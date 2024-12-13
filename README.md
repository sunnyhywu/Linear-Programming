# Linear-Programming
12/13

Add W, h_k, T_k, q_k 

In x_1,x_2,x_3, all coefficients are right.

但其他情況我還要再試一下。


12/12
Modify矩陣數字的順序+學妹加上去的版本 in old_version.cs

[x_1,x_2,x_3,y_1_1,y_2_1,y_1_2,y_2_2,y_1_3,y_2_3,w_1_1,w_2_1,w_3_1,w_4_1,w_1_2,w_2_2,w_3_2,w_4_2,w_1_3,w_2_3,w_3_3,w_4_3]

-30x1-y11+w11<=-200
-3.6x2-y21+w21<=-240
-24x3+w31+w41<=0
w31<=6000


12/11
Add dual funaction and have duals in SolveUsingPrimalSimplexSubproblem

12/7
我目前完成
let user choose:

1.how much scenarios use want to do

2.which variables to enter to master problem, can be x, y, w

格式：x_1,y_1_2,w_1_2

3.先把所有的目標函式係數和限制式讀成objectiveCoefficients、lhsMatrix、rhsMatrix，因為求解會用到之前的simplex function，所以把限制式全部改成<=的方向，有些係數有變號。

矩陣數字的順序：[x_1,x_2,x_3, w_1_1,w_2_1,w_3_1,w_4_1,y_1_1,y_2_1 ......]


4.再根據輸入的master variable形成不同的masterCoefficients, subproblemCoefficients, masterLhs, masterRhs, subproblemLhs, subproblemRhs

master的限制式的選取方式是那個不等式的係數master variable都有pick那就算master限制式
以簡報為例，如果我選x_1,x_2,x_3為master variable，則master限制式只有x_1+x_2+x_3<=500，其他都算subproblem的

masterCoefficients和subproblemCoefficients也有加slack variable項進去
masterLhs和subproblemLhs有加 slack variable項進去