# COCOB-Implementation

# Description

In coinBetting.py:
1. The function "coin_betting" implements a 1D Coin Betting through ONS, with v_0 = -0.5.
2. The function "cocob_torch" implements a n-Dimension Coin Betting through ONS, with each coordinate's v_0 = -0.5.
3. The function "adagrad" implements AdaGrad algorithm.

In oracles.py:
1. The function "real_coin_value1" returns the subgradient for 1D function f(x) = |x-0.2|
2. The function "real_coin_value2" returns the subgradient for 1D function f(x) = 1/8*(x-10)^2
2. The function "real_coin_value3" returns the subgradient for 2D function f(x) = sqrt(x.T * A * x) 
   where A = [[0.2,0.0],[0.0,0.5]].
   

In test.py:
1. The updated v_t in COCOB-ONS is stored in v_list and visualized in plotting figure(1) and figure(2).
2. The comparison between AdaGrad and COCOB-ONS is also visualized.

In test2.py:
1. Test COCOB-ONS using 2D function f(x) of coinBetting.func1_grad
2. The function "draw_contour" is used to visualize the updates of COCOB-ONS and AdaGrad on the 2D plane.
