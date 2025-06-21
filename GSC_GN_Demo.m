%% Geometric Shape Characterization-based Algorithm for Electrical Impedance Tomography Reconstruction
% Gauss-Newton Algorithm
% ============= Author =============
% Zekun Chen (chenzk806@nenu.edu.cn, 229800776@qq.com)
% ============= Method Description =============
% 1. Parameterization:
%    - The target's boundary is described by a closed cubic B-spline curve.
%    - The optimization variables are a combined vector:
%      - Radial distances 'd' for each control point.
%      - The target (inclusion) conductivity 'sigma_in'.
%      - The background conductivity 'sigma_out'.
%
% 2. Optimization Algorithm (Gauss-Newton Method):
%    - The algorithm iteratively refines the shape and conductivities to minimize
%      a cost function comprising four terms: data fidelity, inclusion and background
%      conductivity regularization, and shape smoothness regularization.
%      Cost = ||V_comp - V_meas||^2 + alpha_i*(sigma_i - sigma_i*)^2 + alpha_b*(sigma_b - sigma_b*)^2 + alpha_g*||L*d||^2
%
% 3. Optimization Steps (Difference Imaging):
%    - An augmented residual vector and its corresponding Jacobian 'J_aug' are constructed.
%    - The linear equation for the update step 'delta_p' is solved: (J_aug'*J_aug + lambda*I)*delta_p = J_aug'*residual_aug
%    - Parameters are updated: p_new = p_old - learning_rate * delta_p
%
% 4. Real-time Visualization:
%    - The script provides real-time plotting of the B-spline curve's evolution
%      and the estimated conductivities.

%% 0. Initialize EIDORS
clear all; % Clear all variables from the workspace
close all; % Close all open figures

% Users need to modify this line according to their EIDORS installation path
run 'E:\eidors-v3.10-ng\eidors\startup.m';


%% 1. EIDORS Forward Problem Setup
fprintf('Step 1: Setting up EIDORS forward problem model...\n');
n_elecs = 16; % Number of electrodes

% use the 'i2c' model, a common 2D circular model configuration
imdl = mk_common_model('i2c', n_elecs);
if isfield(imdl, 'fwd_model') && ~isempty(imdl.fwd_model)
    fmdl = imdl.fwd_model;
else
    error('mk_common_model did not return a valid imdl struct containing fwd_model.');
end
fprintf('Model created using mk_common_model("i2c", %d).\n', n_elecs);

% Calculate the center point coordinates of each element in the finite element mesh
elem_centroids = [];
if isfield(fmdl, 'nodes') && isfield(fmdl, 'elems')
    try
        elem_centroids = get_elem_centre(fmdl);
        if size(elem_centroids,2) < 2
            error('Insufficient dimensions for element centroid coordinates.');
        end
    catch ME_get_centre
        % If get_elem_centre function fails, calculate manually
        fprintf('Warning: get_elem_centre function failed (%s). Attempting to calculate element centers manually.\n', ME_get_centre.message);
        elem_centroids = zeros(size(fmdl.elems, 1), size(fmdl.nodes,2));
        for i_elem = 1:size(fmdl.elems, 1)
            elem_nodes_indices = fmdl.elems(i_elem, :);
            elem_node_coords = fmdl.nodes(elem_nodes_indices, :);
            elem_centroids(i_elem, :) = mean(elem_node_coords, 1);
        end
        if size(elem_centroids,2) < 2
            fprintf('Error: Insufficient dimensions for manually calculated element centroid coordinates.\n'); return;
        end
    end
else
    fprintf('Error: fmdl struct is incomplete, missing nodes or elems field.\n'); return;
end
if isempty(elem_centroids)
    fprintf('Error: Failed to get element centroid coordinates.\n'); return;
end
elem_coords = elem_centroids(:,1:2);
num_elements = size(elem_coords, 1);
fprintf('Model setup complete. Electrodes: %d, Elements: %d\n', n_elecs, num_elements);

% Load custom colormap
try
    myColorMap = load('CMDR.txt');
    fprintf('Custom colormap CMDR.txt loaded successfully.\n');
catch
    fprintf('Warning: Failed to load custom colormap CMDR.txt, using fallback colormap ''jet''.\n');
    myColorMap = jet(256);
end

%% 2. Measurement Data (V) - Generate Simulated "Difference" Data
fprintf('Step 2: Generating simulated difference measurement data (target is a square)...\n');
background_conductivity_true = 1.0;
conductivity_square_true = 0.1;

% --- First, create a reference image of the homogeneous background and get reference voltages ---
img_homogeneous = mk_image(fmdl, background_conductivity_true);
V_homogeneous = fwd_solve(img_homogeneous).meas;

% --- Second, create the image with the target object ---
true_conductivity_img = mk_image(fmdl, background_conductivity_true);
% Define the parameters of the target square
square_center_x = -0.5;
square_center_y = 0.0;
square_half_side = 0.25;
% Assign conductivity to the elements inside the square
selector = abs(elem_coords(:,1) - square_center_x) < square_half_side & ...
           abs(elem_coords(:,2) - square_center_y) < square_half_side;
true_elements_data = ones(num_elements, 1) * background_conductivity_true;
true_elements_data(selector) = conductivity_square_true;
true_conductivity_img.elem_data = true_elements_data;
V_with_object = fwd_solve(true_conductivity_img).meas;

% --- Third, calculate the difference voltage and add noise ---
snr_db = 60;
V_diff_true = V_with_object - V_homogeneous;
noise = randn(size(V_diff_true)) * std(V_diff_true) / (10^(snr_db/20));
V_measured_diff = V_diff_true + noise;

fprintf('Simulated difference data generated. Measurement vector size: %s\n', mat2str(size(V_measured_diff)));

% Visualize the true conductivity and difference voltage
figure('Name','True Conductivity and Difference Voltage','NumberTitle','off');
subplot(1,2,1); show_fem_with_custom_cmap(true_conductivity_img, myColorMap); title('True Conductivity (Single Square)');
subplot(1,2,2); plot(V_measured_diff); title('Simulated Difference Boundary Voltage');
drawnow;

% =========================================================================
% ======================= OPTIMIZATION PROCESS STARTS HERE ========================
% =========================================================================

%% 3. B-Spline Curve Parameter Setup
fprintf('Step 3: Setting up B-Spline curve parameters...\n');
num_cp = 15; % Number of control points for the B-spline curve
spline_degree = 3; % Degree of the B-spline curve (cubic)
num_params_geom = num_cp;
num_params = num_cp + 2; % Total parameters: distances + 2 conductivities

% Define a fixed center and the angle for each control point
ini_center_x = -0.45;
ini_center_y = 0.01;
center_coords = [ini_center_x, ini_center_y];
theta_step = 2 * pi / num_cp;
fixed_angles = ((1:num_cp)' .* theta_step) - (theta_step / 2);
unit_vectors = [sin(fixed_angles), cos(fixed_angles)]; % Unit direction vector for each control point

fprintf('B-Spline setup complete: %d control points.\n', num_cp);
fprintf('Total optimization parameters = %d (%d distances + 2 conductivities)\n', num_params, num_cp);

%% 4. Optimization Parameter Setup
fprintf('Step 4: Defining optimization parameters...\n');
max_iterations = 15;
learning_rate = 0.4;   % Update step size for the Gauss-Newton method
lambda_gn = 1e-2;      % Tikhonov regularization parameter for GN step
fd_step = 1e-4;        % Step size for finite difference Jacobian calculation
epsilon_smooth = 0.03; % Thickness parameter for smoothing the boundary

% Regularization weights from the cost function
alpha_sigma_incl = 0.5; % Regularization for inclusion conductivity
alpha_sigma_bg = 0.1;   % Regularization for background conductivity
alpha_geom = 1e-3;      % Regularization for shape smoothness (1st derivative)

% Parameter bounds
min_dist = 0.05;  max_dist = 0.7;
min_sigma = 0.01; max_sigma = 2.0;

fprintf('Gauss-Newton regularization lambda: %e\n', lambda_gn);
fprintf('Inclusion conductivity regularization alpha: %.2e\n', alpha_sigma_incl);
fprintf('Background conductivity regularization alpha: %.2e\n', alpha_sigma_bg);
fprintf('Geometry smoothness regularization alpha: %.2e\n', alpha_geom);

%% 5. Optimization Loop (Gauss-Newton Method)
fprintf('\n===== Starting Optimization: Gauss-Newton with Full Cost Function =====\n');

% Initialize the combined parameter vector p = [distances; sigma_in; sigma_out]
initial_distance = 0.05;
initial_sigma_in = 1.0;  % Initial guess for inclusion conductivity
initial_sigma_out = 1.0; % Initial guess for background conductivity
params_vec = [ones(num_cp, 1) * initial_distance; initial_sigma_in; initial_sigma_out];

% Define target values for regularization
target_sigma_incl = conductivity_square_true;
target_sigma_bg = background_conductivity_true;

history_loss = zeros(max_iterations, 1);

% Construct shape smoothing regularization matrix (1st derivative operator)
L = spdiags([-ones(num_cp, 1), ones(num_cp, 1)], [-1, 0], num_cp, num_cp);
L(1, num_cp) = -1; % Enforce wrap-around for the closed curve

% Iteration loop
for iter = 1:max_iterations
    
    % Decompose current parameter vector
    distances_vec = params_vec(1:num_cp);
    sigma_in_current = params_vec(num_cp + 1);
    sigma_out_current = params_vec(num_cp + 2);
    
    % --- 1. Compute current conductivity, voltage, and augmented residual ---
    [sigma_current_vec, ~, H_smooth] = generate_sigma_from_b_spline_curve_smooth(center_coords + distances_vec .* unit_vectors, elem_coords, sigma_in_current, sigma_out_current, spline_degree, epsilon_smooth);
    img_current = mk_image(fmdl, sigma_current_vec);
    V_computed_abs = fwd_solve(img_current).meas;
    V_computed_diff = V_computed_abs - V_homogeneous;
    
    % Assemble the augmented residual vector from all 4 cost function terms
    residual_V = V_computed_diff - V_measured_diff;
    residual_sigma_incl = sqrt(alpha_sigma_incl) * (sigma_in_current - target_sigma_incl);
    residual_sigma_bg = sqrt(alpha_sigma_bg) * (sigma_out_current - target_sigma_bg);
    residual_geom = sqrt(alpha_geom) * (L * distances_vec);
    residual_aug = [residual_V; residual_sigma_incl; residual_sigma_bg; residual_geom];
    current_loss = sum(residual_aug.^2);
    history_loss(iter) = current_loss;

    fprintf('Iter %d/%d: Loss: %.3e, sig_in: %.3f, sig_out: %.3f\n', ...
        iter, max_iterations, current_loss, sigma_in_current, sigma_out_current);
    
    % --- 2. Compute Augmented Jacobian Matrix J_aug ---
    
    % --- 2a. Standard EIT Jacobian J_sigma (dV/d_sigma) ---
    J_sigma = calc_jacobian(img_current);
    
    % --- 2b. Parameter Jacobian J_p (d_sigma/dp) via finite differences and chain rule ---
    J_p = zeros(num_elements, num_params);
    % Part 1: Derivatives w.r.t. control point distances (d_sigma/d_dist)
    for j = 1:num_cp
        distances_perturbed = distances_vec;
        distances_perturbed(j) = distances_perturbed(j) + fd_step;
        sigma_perturbed_vec = generate_sigma_from_b_spline_curve_smooth(center_coords + distances_perturbed .* unit_vectors, elem_coords, sigma_in_current, sigma_out_current, spline_degree, epsilon_smooth);
        J_p(:, j) = (sigma_perturbed_vec - sigma_current_vec) / fd_step;
    end
    % Part 2: Derivatives w.r.t. conductivities (d_sigma/d_sigma_in, d_sigma/d_sigma_out)
    J_p(:, num_cp + 1) = H_smooth;      % Derivative w.r.t. sigma_in
    J_p(:, num_cp + 2) = 1 - H_smooth;  % Derivative w.r.t. sigma_out

    % --- 2c. Assemble the full Augmented Jacobian ---
    J_V_part = J_sigma * J_p; % Data-term Jacobian part
    J_sigma_reg_part = zeros(2, num_params);
    J_sigma_reg_part(1, num_params-1) = sqrt(alpha_sigma_incl);
    J_sigma_reg_part(2, num_params)   = sqrt(alpha_sigma_bg);
    J_geom_reg_part = [sqrt(alpha_geom) * L, zeros(num_params_geom, 2)]; % Pad with zeros for conductivity params
    
    J_aug = [J_V_part; J_sigma_reg_part; J_geom_reg_part];

    % --- 3. Solve for the Gauss-Newton update step ---
    hessian_approx = J_aug' * J_aug;
    gradient_approx = J_aug' * residual_aug;
    regularization_matrix = lambda_gn * eye(num_params);
    delta_p = (hessian_approx + regularization_matrix) \ gradient_approx;
    
    % --- 4. Update the parameter vector ---
    params_vec = params_vec - learning_rate * delta_p;
    
    % --- 5. Apply boundary constraints ---
    params_vec(1:num_cp) = max(min(params_vec(1:num_cp), max_dist), min_dist);
    params_vec(num_cp + 1) = max(min(params_vec(num_cp + 1), max_sigma), min_sigma);
    params_vec(num_cp + 2) = max(min(params_vec(num_cp + 2), max_sigma), min_sigma);
    
    % --- 6. Real-time visualization ---
    plot_optimization_progress_gn(params_vec(1:num_cp), params_vec(num_cp+1), params_vec(end), iter, current_loss, unit_vectors, center_coords, fmdl, elem_coords, spline_degree, epsilon_smooth, myColorMap);
end

fprintf('Optimization finished.\n');

%% 6. Final Result Evaluation and Visualization
fprintf('Step 6: Evaluating and visualizing final results...\n');
final_distances = params_vec(1:num_cp);
final_sigma_in = params_vec(num_cp+1);
final_sigma_out = params_vec(num_cp+2);

fprintf('Final Estimated Inclusion Conductivity: %.4f (True: %.4f)\n', final_sigma_in, conductivity_square_true);
fprintf('Final Estimated Background Conductivity: %.4f (True: %.4f)\n', final_sigma_out, background_conductivity_true);

% Generate final image from optimized parameters
[final_sigma, final_boundary_coords] = generate_sigma_from_b_spline_curve_smooth(...
    center_coords + final_distances .* unit_vectors, elem_coords, ...
    final_sigma_in, final_sigma_out, spline_degree, epsilon_smooth);
final_reconstruction_img = mk_image(fmdl, final_sigma);

figure('Name','Final EIT Reconstruction Result (Gauss-Newton)','NumberTitle','off', 'Position', [950, 300, 1000, 400]);
subplot(1,2,1);
show_fem_with_custom_cmap(true_conductivity_img, myColorMap);
title(sprintf('True Conductivity\nTarget: %.2f, Bkg: %.2f', conductivity_square_true, background_conductivity_true));

subplot(1,2,2);
show_fem_with_custom_cmap(final_reconstruction_img, myColorMap);
hold on;
plot(final_boundary_coords(:,1), final_boundary_coords(:,2), 'k-', 'LineWidth', 2);
hold off;
title(sprintf('Final Reconstruction (GN)\nTarget: %.2f, Bkg: %.2f', final_sigma_in, final_sigma_out));

disp('Script execution completed.');
disp('====================================================');


%% Helper Functions

function plot_optimization_progress_gn(distances, sigma_in, sigma_out, iter, loss, unit_vectors, center_coords, fmdl, elem_coords, degree, epsilon, myColorMap)
    % This function visualizes the state of the reconstruction at each iteration.
    current_cp_coords = center_coords + distances .* unit_vectors;
    [sigma_current, boundary_poly_coords] = generate_sigma_from_b_spline_curve_smooth(current_cp_coords, elem_coords, sigma_in, sigma_out, degree, epsilon);
    current_sigma_img = mk_image(fmdl, sigma_current);
    
    fig_handle_recon = findobj('type','figure','name','EIT Real-time Reconstruction (GN)');
    if isempty(fig_handle_recon); fig_handle_recon = figure('Name','EIT Real-time Reconstruction (GN)', 'NumberTitle','off', 'Position', [200, 300, 600, 500]); end
    figure(fig_handle_recon);
    
    show_fem_with_custom_cmap(current_sigma_img, myColorMap);
    hold on;
    plot(boundary_poly_coords(:,1), boundary_poly_coords(:,2), 'k-', 'LineWidth', 2);
    plot(current_cp_coords(:,1), current_cp_coords(:,2), 'ro-', 'LineWidth', 1.5, 'MarkerFaceColor','r');
    hold off;
    title_str = sprintf('GN Iter %d | Loss: %.3e\n{\\sigma}_{in}: %.3f, {\\sigma}_{out}: %.3f', iter, loss, sigma_in, sigma_out);
    title(title_str);
    drawnow;
end

function show_fem_with_custom_cmap(img, custom_cmap)
    show_fem(img);
    colorbar;
    colormap(gca, custom_cmap);
    axis equal;
    axis off;
end

function [sigma_dist, boundary_poly_coords, H_smooth] = generate_sigma_from_b_spline_curve_smooth(cp_coords, elem_centroids, sigma_in, sigma_out, degree, epsilon)
    num_eval_points = 200;
    boundary_poly_coords = eval_closed_bspline_curve(cp_coords, degree, num_eval_points);
    signed_dist = signed_distance_to_polygon(elem_centroids, boundary_poly_coords);
    k = 2 / epsilon;
    H_smooth = 1 ./ (1 + exp(k * signed_dist));
    sigma_dist = sigma_out + (sigma_in - sigma_out) * H_smooth;
end

function sd = signed_distance_to_polygon(points, poly_verts)
    [in, ~] = inpolygon(points(:,1), points(:,2), poly_verts(:,1), poly_verts(:,2));
    ud = zeros(size(points,1),1);
    for i=1:size(points,1)
        dists_to_segments_sq = dist_point_to_segment_sq(points(i,:), poly_verts, [poly_verts(2:end,:); poly_verts(1,:)]);
        ud(i) = sqrt(min(dists_to_segments_sq));
    end
    sd = ud;
    sd(in) = -sd(in);
end

function dist_sq = dist_point_to_segment_sq(p, v, w)
    l2 = sum((v - w).^2, 2);
    l2_zero_idx = l2 < eps;
    l2(l2_zero_idx) = 1; 
    t = sum((p - v) .* (w - v), 2) ./ l2;
    t = max(0, min(1, t));
    projection = v + t .* (w - v);
    dist_sq = sum((p - projection).^2, 2);
    dist_sq(l2_zero_idx) = sum((p-v(l2_zero_idx,:)).^2,2);
end

function points = eval_closed_bspline_curve(cp_coords, degree, num_eval_points)
    points = de_boor_eval(cp_coords, degree, num_eval_points);
end

function points = de_boor_eval(cp_coords, degree, num_eval_points)
    N = size(cp_coords, 1);
    if N <= degree
        points = repmat(mean(cp_coords,1), num_eval_points, 1);
        if isempty(points); points = zeros(num_eval_points, 2); end
        return;
    end
    
    wrapped_cp = [cp_coords(end-degree+1:end, :); cp_coords; cp_coords(1:degree, :)];
    num_wrapped_cp = size(wrapped_cp, 1);
    knot_vector = 0:(num_wrapped_cp + degree);
    
    t_start = degree;
    t_end = N + degree;
    t_eval = linspace(t_start, t_end, num_eval_points);
    points = zeros(num_eval_points, 2);
    
    for i = 1:num_eval_points
        t = t_eval(i);
        k = find(knot_vector <= t, 1, 'last');
        if k > num_wrapped_cp; k = num_wrapped_cp; end
        
        d = wrapped_cp(k-degree:k, :);
        
        for r = 1:degree
            for j = degree:-1:r
                knot_denom = (knot_vector(j+1+k-r) - knot_vector(j+k-degree));
                if abs(knot_denom) < eps
                    alpha = 0;
                else
                    alpha = (t - knot_vector(j+k-degree)) / knot_denom;
                end
                d(j+1,:) = (1-alpha)*d(j,:) + alpha*d(j+1,:);
            end
        end
        points(i, :) = d(degree+1, :);
    end
end
