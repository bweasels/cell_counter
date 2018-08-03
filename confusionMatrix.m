%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: confusionMatrix.m
% Purpose: Loads the training Set
% Notes: This is my first time!!!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [TruePos, TrueNeg, FalsePos, FalseNeg] = confusionMatrix(Y, predict)
TruePos = 0;
TrueNeg = 0;
FalsePos = 0;
FalseNeg = 0;
  for i = 1:length(Y)
    if (Y(i) == predict(i) && Y(i) == 1)
      TruePos = TruePos+1;
    elseif (Y(i) == predict(i) && Y(i) == 0)
      TrueNeg = TrueNeg+1;
    elseif (Y(i) != predict(i) && Y(i) == 1)
      FalseNeg = FalseNeg+1;
    elseif (Y(i) != predict(i) && Y(i) == 0)
      FalsePos = FalsePos+1;
    endif
  endfor
endfunction
