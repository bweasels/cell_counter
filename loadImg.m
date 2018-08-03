%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: loadImg.m
% Purpose: Loads the image, mean centers and flattens
% Notes: This is my first time!!!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [flat] = loadImg(imgAddr)
   image = double(imread(imgAddr));
   %image = imresize(image, [20 20]);
   flat = image(:);
   %flat = flat - mean(flat);
   %flat = flat/std(flat);
end