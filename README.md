# Linear-Programming
## framework (2024/12/10 ver.)

### part 1: input the problem
#### read the problem and let user enter the scenario and the selectesd variable >> done

### part 2: L-shape method
#### solve the all scenarios in one iteration, the stop condition is the theta are close to the last one
#### step in each iteration:
#### 1. solve the master-problem, use the primal to solve it for the first iteration. for the continue, try the last iteration's solution as the initial solution and solve it by the dual. if for the dual, the last solution is infeasible, stop and re-solve it by the primal. if it's feasible, use dual to find the optimal solution.
#### 2. test the feasibility for each scenario, if there is any scenario fail the feasible test, do the feasibility cut and go back to step 1. add the feasibility cut into the master problem and solve it again. if the scenario is feasible, get the optimal solution as 'w' (cost for each scenario, get it by using primal to solve the subproblem).
#### 3. at step 3, we make sure that all the scenarios are feasible. test whether the θ in this iteration is close enough to the last iteration. (*problem: how to know is close or not?) θ is negative infinity in the first iteration. how to calculate θ: ![image](https://github.com/user-attachments/assets/682da0e7-f5af-4a39-977a-1b5f1a1fc644) if the θ are close enough, stop doing any iteration and can get the final cost by adding the θ and the cost of the master problem. if not close enough, go to step 4.
#### 4. find the optimality cut by doing the dual, then add it into master problem. re-solve the master problem by primal, using the now solution as the initial solution. after solve it, go to step 2.

### part 3: get the final solution
#### take the final iteration θ and add it to the final master problem's optimal solution, which represent the cost for the master problem.


