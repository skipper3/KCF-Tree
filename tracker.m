function [positions, time] = tracker(video_path, img_files, pos, target_sz, ...
    padding, kernel, lambda, output_sigma_factor, interp_factor, cell_size, ...
    features, show_visualization)
%TRACKER Kernelized/Dual Correlation Filter (KCF/DCF) tracking.
%   This function implements the pipeline for tracking with the KCF (by
%   choosing a non-linear kernel) and DCF (by choosing a linear kernel).
%
%   It is meant to be called by the interface function RUN_TRACKER, which
%   sets up the parameters and loads the video information.
%
%   Parameters:
%     VIDEO_PATH is the location of the image files (must end with a slash
%      '/' or '\').
%     IMG_FILES is a cell array of image file names.
%     POS and TARGET_SZ are the initial position and size of the target
%      (both in format [rows, columns]).
%     PADDING is the additional tracked region, for context, relative to
%      the target size.
%     KERNEL is a struct describing the kernel. The field TYPE must be one
%      of 'gaussian', 'polynomial' or 'linear'. The optional fields SIGMA,
%      POLY_A and POLY_B are the parameters for the Gaussian and Polynomial
%      kernels.
%     OUTPUT_SIGMA_FACTOR is the spatial bandwidth of the regression
%      target, relative to the target size.
%     INTERP_FACTOR is the adaptation rate of the tracker.
%     CELL_SIZE is the number of pixels per cell (must be 1 if using raw
%      pixels).
%     FEATURES is a struct describing the used features (see GET_FEATURES).
%     SHOW_VISUALIZATION will show an interactive video if set to true.
%
%   Outputs:
%    POSITIONS is an Nx2 matrix of target positions over time (in the
%     format [rows, columns]).
%    TIME is the tracker execution time, without video loading/rendering.
%
%   Joao F. Henriques, 2014


%if the target is large, lower the resolution, we don't need that much
%detail
resize_image = (sqrt(prod(target_sz)) >= 100);  %diagonal size >= threshold
if resize_image,
    pos = floor(pos / 2);
    target_sz = floor(target_sz / 2);
end


%window size, taking padding into account
window_sz = floor(target_sz * (1 + padding));

% 	%we could choose a size that is a power of two, for better FFT
% 	%performance. in practice it is slower, due to the larger window size.
% 	window_sz = 2 .^ nextpow2(window_sz);


%create regression labels, gaussian shaped, with a bandwidth
%proportional to target size
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));

%store pre-computed cosine window
cos_window = hann(size(yf,1)) * hann(size(yf,2))';


if show_visualization,  %create video interface
    update_visualization = show_video(img_files, video_path, resize_image);
end


%note: variables ending with 'f' are in the Fourier domain.

time = 0;  %to calculate FPS
positions = zeros(numel(img_files), 2);  %to calculate precision

for frame = 1:numel(img_files),
    %load image
    im = imread([video_path img_files{frame}]);
    if size(im,3) > 1,
        im = rgb2gray(im);
    end
    if resize_image,
        im = imresize(im, 0.5);
    end
    
    tic()
    if frame == 1,  %first frame, train with a single image
        patch = get_subwindow(im, pos, window_sz);
        feat=get_features(patch, features, cell_size, cos_window);
        xf = fft2(feat);
        
        %Kernel Ridge Regression, calculate alphas (in Fourier domain)
        switch kernel.type
            case 'gaussian',
                kf = gaussian_correlation(xf, xf, kernel.sigma);
            case 'polynomial',
                kf = polynomial_correlation(xf, xf, kernel.poly_a, kernel.poly_b);
            case 'linear',
                kf = linear_correlation(xf, xf);
        end
        alphaf = yf ./ (kf + lambda);
        model_alphaf = alphaf;
        model_xf = xf;
        father_node=0;
        father(1)=1;
        %subsequent frames, interpolate model
        temp=cell(1,2);
        temp{1,1}=model_alphaf;
        temp{1,2}=model_xf;
        t=tree(temp);
        tree_index=1;
        weight(1)=1;
        
    end
    
    if frame > 1,
        %obtain a subwindow for detection at the position from last
        %frame, and convert to Fourier domain (its size is unchanged)
        sum_temp=0;
        response_temp=0;
        
        patch = get_subwindow(im, pos, window_sz);
        feat=get_features(patch, features, cell_size, cos_window);
        temp=t.get(tree_index);
        model_alphaf=temp{1,1};
        model_xf=temp{1,2};
        %father_alphaf=t.get(father(i));
        zf = fft2(feat);
        
        %calculate response of the classifier at all shifts
        switch kernel.type
            case 'gaussian',
                kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
            case 'polynomial',
                kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);
            case 'linear',
                kzf = linear_correlation(zf, model_xf);
        end
        response = real(fftshift(ifft2(model_alphaf .* kzf)));
        if max(response(:))>0.5
            [vert_delta, horiz_delta] = find(response == max(response(:)), 1);
            matrix_r=vert_delta;
            matrix_c=horiz_delta;
            vert_delta  = vert_delta  - floor(size(zf,1)/2);
            horiz_delta = horiz_delta - floor(size(zf,2)/2);
            pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];
        else if tree_index<2
                
                
                
                for i=1:tree_index
                    temp=t.get(i);
                    model_alphaf=temp{1,1};
                    model_xf=temp{1,2};
                    %father_alphaf=t.get(father(i));
                    zf = fft2(feat);
                    
                    %calculate response of the classifier at all shifts
                    switch kernel.type
                        case 'gaussian',
                            kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
                        case 'polynomial',
                            kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);
                        case 'linear',
                            kzf = linear_correlation(zf, model_xf);
                    end
                    response = real(fftshift(ifft2(model_alphaf .* kzf)));%equation for fast detection
                    figure(100);
                    subplot(1,2,i);
                    imagesc(response);
                    
                    
                    psr  = PSR( response, 0.15 );
                    %father_psr  = PSR( father_response, 0.15 );
                    weight(i)=psr;
                    %father_weight(father(i))=father_psr;
                    response = response*weight(i);
                    %father_response=father_response*father_weight(father(i));
                    sum_temp=sum_temp+weight(i);
                    
                    response_temp=response_temp+response;
                end
                %target location is at the maximum response. we must take into
                %account the fact that, if the target doesn't move, the peak
                %will appear at the top-left corner, not at the center (this is
                %discussed in the paper). the responses wrap around cyclically.
                
                response_temp=response_temp/sum_temp;
                
                [vert_delta, horiz_delta] = find(response_temp == max(response_temp(:)), 1);
                matrix_r=vert_delta;
                matrix_c=horiz_delta;
                vert_delta  = vert_delta  - floor(size(zf,1)/2);
                horiz_delta = horiz_delta - floor(size(zf,2)/2);
                pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];
                
            else for i=tree_index-1:tree_index
                    temp=t.get(i);
                    model_alphaf=temp{1,1};
                    model_xf=temp{1,2};
                    zf = fft2(feat);
                    
                    %calculate response of the classifier at all shifts
                    switch kernel.type
                        case 'gaussian',
                            kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
                        case 'polynomial',
                            kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);
                        case 'linear',
                            kzf = linear_correlation(zf, model_xf);
                    end
                    response = real(fftshift(ifft2(model_alphaf .* kzf)));  %equation for fast detection
                    figure(100);
                    subplot(1,2,i-tree_index+2);
                    imagesc(response);
                    
                    psr  = PSR( response, 0.15 );
                    
                    weight(i)=psr;
                    
                    response = response*weight(i);
                    
                    sum_temp=sum_temp+weight(i);
                    
                    response_temp=response_temp+response;
                end
                
                response_temp=response_temp/sum_temp;
                
                [vert_delta, horiz_delta] = find(response_temp == max(response_temp(:)), 1);
                matrix_r=vert_delta;
                matrix_c=horiz_delta;
                vert_delta  = vert_delta  - floor(size(zf,1)/2);
                horiz_delta = horiz_delta - floor(size(zf,2)/2);
                pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];
            end
        end
        if mod(frame,50)==0
            
            
            
            %             patch = get_subwindow(im, pos, window_sz);
            %             feat=get_features(patch, features, cell_size, cos_window);
            %             xf = fft2(feat);
            %
            %             %Kernel Ridge Regression, calculate alphas (in Fourier domain)
            %             switch kernel.type
            %                 case 'gaussian',
            %                     kf = gaussian_correlation(xf, xf, kernel.sigma);
            %                 case 'polynomial',
            %                     kf = polynomial_correlation(xf, xf, kernel.poly_a, kernel.poly_b);
            %                 case 'linear',
            %                     kf = linear_correlation(xf, xf);
            %             end
            %             alphaf = yf ./ (kf + lambda);
            %             model_alphaf = alphaf;
            
            
            %model_alphaf=t.get(tree_index);%取上一个tracker索引的参数作为下一个索引的参数
            tree_index=tree_index+1;%每10帧生成一个新的tracker
            %%
            %寻找tracker的父节点
            max_temp=0;
            
            %judge edge between all tracker
            %找出之前tracker中对当前帧response最大的作为父节点
            
            for i=1:tree_index-1
                temp=t.get(i);
                temp_alphaf=temp{1,1};
                temp_xf=temp{1,2};
                zf = fft2(feat);
                
                %calculate response of the classifier at all shifts
                switch kernel.type
                    case 'gaussian',
                        kzf = gaussian_correlation(zf, temp_xf, kernel.sigma);
                    case 'polynomial',
                        kzf = polynomial_correlation(zf, temp_xf, kernel.poly_a, kernel.poly_b);
                    case 'linear',
                        kzf = linear_correlation(zf, temp_xf);
                end
                response = real(fftshift(ifft2(temp_alphaf .* kzf)));
                if response(matrix_r,matrix_c)>=max_temp
                    max_temp=response(matrix_r,matrix_c);
                    father_node=i;
                end
            end
            
            t=t.addnode(father_node,t.get(tree_index-1));
            father(tree_index)=father_node;
            
        end
        
        %obtain a subwindow for training at newly estimated target position
        
        patch = get_subwindow(im, pos, window_sz);
        
        feat=get_features(patch, features, cell_size, cos_window);
        xf = fft2(feat);
        
        %Kernel Ridge Regression, calculate alphas (in Fourier domain)
        switch kernel.type
            case 'gaussian',
                kf = gaussian_correlation(xf, xf, kernel.sigma);
            case 'polynomial',
                kf = polynomial_correlation(xf, xf, kernel.poly_a, kernel.poly_b);
            case 'linear',
                kf = linear_correlation(xf, xf);
        end
        alphaf = yf ./ (kf + lambda);   %equation for fast training
        temp=t.get(floor(frame/50)+1);
        model_alphaf=temp{1,1};
        model_xf=temp{1,2};
        
        model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
        
        model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
        temp{1,1}=model_alphaf;
        temp{1,2}=model_xf;
        t=t.set(floor(frame/50)+1,temp);
    end
    %save position and timing
    positions(frame,:) = pos;
    time = time + toc();
    
    %visualization
    if show_visualization,
        box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        stop = update_visualization(frame, box);
        if stop, break, end  %user pressed Esc, stop early
        
        drawnow
        % 			pause(0.05)  %uncomment to run slower
    end
    
end

if resize_image,
    positions = positions * 2;
end
%disp(t.tostring);
%weight
end



function [ psr ] = PSR( response, rate )
%PSR Summary of this function goes here
%   Detailed explanation goes here
maxresponse = max(response(:));
%calculate the PSR
range = ceil(sqrt(numel(response))*rate);
response=fftshift(response);
[xx, yy] = find(response == maxresponse, 1);
idx = xx-range:xx+range;
idy = yy-range:yy+range;
idy(idy<1)=1;idx(idx<1)=1;
idy(idy>size(response,2))=size(response,2);idx(idx>size(response,1))=size(response,1);
response(idx,idy)=0;
m = sum(response(:))/numel(response);
d=sqrt(size(response(:),1)*var(response(:))/numel(response));
psr =(maxresponse - m)/d ;

end
