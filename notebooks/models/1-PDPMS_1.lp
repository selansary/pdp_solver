\ Model 1-PDPMS
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  50 x[1,2] + 98 x[1,3] + 54 x[1,4] + 6 x[1,5] + 34 x[1,6] + 66 x[1,7]
   + 63 x[1,8] + 52 x[1,9] + 39 x[1,10] + 62 x[2,1] + 46 x[2,3] + 75 x[2,4]
   + 28 x[2,5] + 65 x[2,6] + 18 x[2,7] + 37 x[2,8] + 18 x[2,9] + 97 x[2,10]
   + 13 x[3,1] + 80 x[3,2] + 33 x[3,4] + 69 x[3,5] + 91 x[3,6] + 78 x[3,7]
   + 19 x[3,8] + 40 x[3,9] + 13 x[3,10] + 94 x[4,1] + 10 x[4,2] + 88 x[4,3]
   + 43 x[4,5] + 61 x[4,6] + 72 x[4,7] + 13 x[4,8] + 46 x[4,9] + 56 x[4,10]
   + 41 x[5,1] + 79 x[5,2] + 82 x[5,3] + 27 x[5,4] + 71 x[5,6] + 62 x[5,7]
   + 57 x[5,8] + 67 x[5,9] + 34 x[5,10] + 8 x[6,1] + 71 x[6,2] + 2 x[6,3]
   + 12 x[6,4] + 93 x[6,5] + 52 x[6,7] + 91 x[6,8] + 86 x[6,9] + 81 x[6,10]
   + x[7,1] + 79 x[7,2] + 64 x[7,3] + 43 x[7,4] + 32 x[7,5] + 94 x[7,6]
   + 42 x[7,8] + 91 x[7,9] + 9 x[7,10] + 25 x[8,1] + 73 x[8,2] + 29 x[8,3]
   + 31 x[8,4] + 19 x[8,5] + 70 x[8,6] + 58 x[8,7] + 12 x[8,9] + 11 x[8,10]
   + 41 x[9,1] + 66 x[9,2] + 63 x[9,3] + 14 x[9,4] + 39 x[9,5] + 71 x[9,6]
   + 38 x[9,7] + 91 x[9,8] + 16 x[9,10] + 71 x[10,1] + 43 x[10,2]
   + 70 x[10,3] + 27 x[10,4] + 78 x[10,5] + 71 x[10,6] + 76 x[10,7]
   + 37 x[10,8] + 57 x[10,9] + 0 y[0,0] + 0 y[0,1] + 0 y[0,2] + 0 y[6,0]
   + 0 y[6,1] + 0 y[6,2] + 0 y[7,0] + 0 y[7,1] + 0 y[7,2] + 0 y[8,0]
   + 0 y[8,1] + 0 y[8,2] + 0 y[9,0] + 0 y[9,1] + 0 y[9,2] + 0 y[10,0]
   + 0 y[10,1] + 0 y[10,2]
Subject To
 out_vertex_constr[0]: x[0,0] + x[0,1] + x[0,2] + x[0,3] + x[0,4] + x[0,5]
   + x[0,6] + x[0,7] + x[0,8] + x[0,9] + x[0,10] = 1
 out_vertex_constr[1]: x[1,0] + x[1,1] + x[1,2] + x[1,3] + x[1,4] + x[1,5]
   + x[1,6] + x[1,7] + x[1,8] + x[1,9] + x[1,10] = 1
 out_vertex_constr[2]: x[2,0] + x[2,1] + x[2,2] + x[2,3] + x[2,4] + x[2,5]
   + x[2,6] + x[2,7] + x[2,8] + x[2,9] + x[2,10] = 1
 out_vertex_constr[3]: x[3,0] + x[3,1] + x[3,2] + x[3,3] + x[3,4] + x[3,5]
   + x[3,6] + x[3,7] + x[3,8] + x[3,9] + x[3,10] = 1
 out_vertex_constr[4]: x[4,0] + x[4,1] + x[4,2] + x[4,3] + x[4,4] + x[4,5]
   + x[4,6] + x[4,7] + x[4,8] + x[4,9] + x[4,10] = 1
 out_vertex_constr[5]: x[5,0] + x[5,1] + x[5,2] + x[5,3] + x[5,4] + x[5,5]
   + x[5,6] + x[5,7] + x[5,8] + x[5,9] + x[5,10] = 1
 out_vertex_constr[6]: x[6,0] + x[6,1] + x[6,2] + x[6,3] + x[6,4] + x[6,5]
   + x[6,6] + x[6,7] + x[6,8] + x[6,9] + x[6,10] = 1
 out_vertex_constr[7]: x[7,0] + x[7,1] + x[7,2] + x[7,3] + x[7,4] + x[7,5]
   + x[7,6] + x[7,7] + x[7,8] + x[7,9] + x[7,10] = 1
 out_vertex_constr[8]: x[8,0] + x[8,1] + x[8,2] + x[8,3] + x[8,4] + x[8,5]
   + x[8,6] + x[8,7] + x[8,8] + x[8,9] + x[8,10] = 1
 out_vertex_constr[9]: x[9,0] + x[9,1] + x[9,2] + x[9,3] + x[9,4] + x[9,5]
   + x[9,6] + x[9,7] + x[9,8] + x[9,9] + x[9,10] = 1
 out_vertex_constr[10]: x[10,0] + x[10,1] + x[10,2] + x[10,3] + x[10,4]
   + x[10,5] + x[10,6] + x[10,7] + x[10,8] + x[10,9] + x[10,10] = 1
 in_vertex_constr[0]: x[0,0] + x[1,0] + x[2,0] + x[3,0] + x[4,0] + x[5,0]
   + x[6,0] + x[7,0] + x[8,0] + x[9,0] + x[10,0] = 1
 in_vertex_constr[1]: x[0,1] + x[1,1] + x[2,1] + x[3,1] + x[4,1] + x[5,1]
   + x[6,1] + x[7,1] + x[8,1] + x[9,1] + x[10,1] = 1
 in_vertex_constr[2]: x[0,2] + x[1,2] + x[2,2] + x[3,2] + x[4,2] + x[5,2]
   + x[6,2] + x[7,2] + x[8,2] + x[9,2] + x[10,2] = 1
 in_vertex_constr[3]: x[0,3] + x[1,3] + x[2,3] + x[3,3] + x[4,3] + x[5,3]
   + x[6,3] + x[7,3] + x[8,3] + x[9,3] + x[10,3] = 1
 in_vertex_constr[4]: x[0,4] + x[1,4] + x[2,4] + x[3,4] + x[4,4] + x[5,4]
   + x[6,4] + x[7,4] + x[8,4] + x[9,4] + x[10,4] = 1
 in_vertex_constr[5]: x[0,5] + x[1,5] + x[2,5] + x[3,5] + x[4,5] + x[5,5]
   + x[6,5] + x[7,5] + x[8,5] + x[9,5] + x[10,5] = 1
 in_vertex_constr[6]: x[0,6] + x[1,6] + x[2,6] + x[3,6] + x[4,6] + x[5,6]
   + x[6,6] + x[7,6] + x[8,6] + x[9,6] + x[10,6] = 1
 in_vertex_constr[7]: x[0,7] + x[1,7] + x[2,7] + x[3,7] + x[4,7] + x[5,7]
   + x[6,7] + x[7,7] + x[8,7] + x[9,7] + x[10,7] = 1
 in_vertex_constr[8]: x[0,8] + x[1,8] + x[2,8] + x[3,8] + x[4,8] + x[5,8]
   + x[6,8] + x[7,8] + x[8,8] + x[9,8] + x[10,8] = 1
 in_vertex_constr[9]: x[0,9] + x[1,9] + x[2,9] + x[3,9] + x[4,9] + x[5,9]
   + x[6,9] + x[7,9] + x[8,9] + x[9,9] + x[10,9] = 1
 in_vertex_constr[10]: x[0,10] + x[1,10] + x[2,10] + x[3,10] + x[4,10]
   + x[5,10] + x[6,10] + x[7,10] + x[8,10] + x[9,10] + x[10,10] = 1
 fulfill_demands_constr[1]: y[1,0] + y[1,1] + y[1,2] = 1
 fulfill_demands_constr[2]: y[2,0] + y[2,1] + y[2,2] = 1
 fulfill_demands_constr[3]: y[3,0] + y[3,1] + y[3,2] = 1
 fulfill_demands_constr[4]: y[4,0] + y[4,1] + y[4,2] = 1
 fulfill_demands_constr[5]: y[5,0] + y[5,1] + y[5,2] = 1
 visiting_order_constr[0,1]: - 10 x[0,1] - u[0] + u[1] >= -9
 visiting_order_constr[0,2]: - 10 x[0,2] - u[0] + u[2] >= -9
 visiting_order_constr[0,3]: - 10 x[0,3] - u[0] + u[3] >= -9
 visiting_order_constr[0,4]: - 10 x[0,4] - u[0] + u[4] >= -9
 visiting_order_constr[0,5]: - 10 x[0,5] - u[0] + u[5] >= -9
 visiting_order_constr[0,6]: - 10 x[0,6] - u[0] + u[6] >= -9
 visiting_order_constr[0,7]: - 10 x[0,7] - u[0] + u[7] >= -9
 visiting_order_constr[0,8]: - 10 x[0,8] - u[0] + u[8] >= -9
 visiting_order_constr[0,9]: - 10 x[0,9] - u[0] + u[9] >= -9
 visiting_order_constr[0,10]: - 10 x[0,10] - u[0] + u[10] >= -9
 visiting_order_constr[1,1]: - 10 x[1,1] >= -9
 visiting_order_constr[1,2]: - 10 x[1,2] - u[1] + u[2] >= -9
 visiting_order_constr[1,3]: - 10 x[1,3] - u[1] + u[3] >= -9
 visiting_order_constr[1,4]: - 10 x[1,4] - u[1] + u[4] >= -9
 visiting_order_constr[1,5]: - 10 x[1,5] - u[1] + u[5] >= -9
 visiting_order_constr[1,6]: - 10 x[1,6] - u[1] + u[6] >= -9
 visiting_order_constr[1,7]: - 10 x[1,7] - u[1] + u[7] >= -9
 visiting_order_constr[1,8]: - 10 x[1,8] - u[1] + u[8] >= -9
 visiting_order_constr[1,9]: - 10 x[1,9] - u[1] + u[9] >= -9
 visiting_order_constr[1,10]: - 10 x[1,10] - u[1] + u[10] >= -9
 visiting_order_constr[2,1]: - 10 x[2,1] + u[1] - u[2] >= -9
 visiting_order_constr[2,2]: - 10 x[2,2] >= -9
 visiting_order_constr[2,3]: - 10 x[2,3] - u[2] + u[3] >= -9
 visiting_order_constr[2,4]: - 10 x[2,4] - u[2] + u[4] >= -9
 visiting_order_constr[2,5]: - 10 x[2,5] - u[2] + u[5] >= -9
 visiting_order_constr[2,6]: - 10 x[2,6] - u[2] + u[6] >= -9
 visiting_order_constr[2,7]: - 10 x[2,7] - u[2] + u[7] >= -9
 visiting_order_constr[2,8]: - 10 x[2,8] - u[2] + u[8] >= -9
 visiting_order_constr[2,9]: - 10 x[2,9] - u[2] + u[9] >= -9
 visiting_order_constr[2,10]: - 10 x[2,10] - u[2] + u[10] >= -9
 visiting_order_constr[3,1]: - 10 x[3,1] + u[1] - u[3] >= -9
 visiting_order_constr[3,2]: - 10 x[3,2] + u[2] - u[3] >= -9
 visiting_order_constr[3,3]: - 10 x[3,3] >= -9
 visiting_order_constr[3,4]: - 10 x[3,4] - u[3] + u[4] >= -9
 visiting_order_constr[3,5]: - 10 x[3,5] - u[3] + u[5] >= -9
 visiting_order_constr[3,6]: - 10 x[3,6] - u[3] + u[6] >= -9
 visiting_order_constr[3,7]: - 10 x[3,7] - u[3] + u[7] >= -9
 visiting_order_constr[3,8]: - 10 x[3,8] - u[3] + u[8] >= -9
 visiting_order_constr[3,9]: - 10 x[3,9] - u[3] + u[9] >= -9
 visiting_order_constr[3,10]: - 10 x[3,10] - u[3] + u[10] >= -9
 visiting_order_constr[4,1]: - 10 x[4,1] + u[1] - u[4] >= -9
 visiting_order_constr[4,2]: - 10 x[4,2] + u[2] - u[4] >= -9
 visiting_order_constr[4,3]: - 10 x[4,3] + u[3] - u[4] >= -9
 visiting_order_constr[4,4]: - 10 x[4,4] >= -9
 visiting_order_constr[4,5]: - 10 x[4,5] - u[4] + u[5] >= -9
 visiting_order_constr[4,6]: - 10 x[4,6] - u[4] + u[6] >= -9
 visiting_order_constr[4,7]: - 10 x[4,7] - u[4] + u[7] >= -9
 visiting_order_constr[4,8]: - 10 x[4,8] - u[4] + u[8] >= -9
 visiting_order_constr[4,9]: - 10 x[4,9] - u[4] + u[9] >= -9
 visiting_order_constr[4,10]: - 10 x[4,10] - u[4] + u[10] >= -9
 visiting_order_constr[5,1]: - 10 x[5,1] + u[1] - u[5] >= -9
 visiting_order_constr[5,2]: - 10 x[5,2] + u[2] - u[5] >= -9
 visiting_order_constr[5,3]: - 10 x[5,3] + u[3] - u[5] >= -9
 visiting_order_constr[5,4]: - 10 x[5,4] + u[4] - u[5] >= -9
 visiting_order_constr[5,5]: - 10 x[5,5] >= -9
 visiting_order_constr[5,6]: - 10 x[5,6] - u[5] + u[6] >= -9
 visiting_order_constr[5,7]: - 10 x[5,7] - u[5] + u[7] >= -9
 visiting_order_constr[5,8]: - 10 x[5,8] - u[5] + u[8] >= -9
 visiting_order_constr[5,9]: - 10 x[5,9] - u[5] + u[9] >= -9
 visiting_order_constr[5,10]: - 10 x[5,10] - u[5] + u[10] >= -9
 visiting_order_constr[6,1]: - 10 x[6,1] + u[1] - u[6] >= -9
 visiting_order_constr[6,2]: - 10 x[6,2] + u[2] - u[6] >= -9
 visiting_order_constr[6,3]: - 10 x[6,3] + u[3] - u[6] >= -9
 visiting_order_constr[6,4]: - 10 x[6,4] + u[4] - u[6] >= -9
 visiting_order_constr[6,5]: - 10 x[6,5] + u[5] - u[6] >= -9
 visiting_order_constr[6,6]: - 10 x[6,6] >= -9
 visiting_order_constr[6,7]: - 10 x[6,7] - u[6] + u[7] >= -9
 visiting_order_constr[6,8]: - 10 x[6,8] - u[6] + u[8] >= -9
 visiting_order_constr[6,9]: - 10 x[6,9] - u[6] + u[9] >= -9
 visiting_order_constr[6,10]: - 10 x[6,10] - u[6] + u[10] >= -9
 visiting_order_constr[7,1]: - 10 x[7,1] + u[1] - u[7] >= -9
 visiting_order_constr[7,2]: - 10 x[7,2] + u[2] - u[7] >= -9
 visiting_order_constr[7,3]: - 10 x[7,3] + u[3] - u[7] >= -9
 visiting_order_constr[7,4]: - 10 x[7,4] + u[4] - u[7] >= -9
 visiting_order_constr[7,5]: - 10 x[7,5] + u[5] - u[7] >= -9
 visiting_order_constr[7,6]: - 10 x[7,6] + u[6] - u[7] >= -9
 visiting_order_constr[7,7]: - 10 x[7,7] >= -9
 visiting_order_constr[7,8]: - 10 x[7,8] - u[7] + u[8] >= -9
 visiting_order_constr[7,9]: - 10 x[7,9] - u[7] + u[9] >= -9
 visiting_order_constr[7,10]: - 10 x[7,10] - u[7] + u[10] >= -9
 visiting_order_constr[8,1]: - 10 x[8,1] + u[1] - u[8] >= -9
 visiting_order_constr[8,2]: - 10 x[8,2] + u[2] - u[8] >= -9
 visiting_order_constr[8,3]: - 10 x[8,3] + u[3] - u[8] >= -9
 visiting_order_constr[8,4]: - 10 x[8,4] + u[4] - u[8] >= -9
 visiting_order_constr[8,5]: - 10 x[8,5] + u[5] - u[8] >= -9
 visiting_order_constr[8,6]: - 10 x[8,6] + u[6] - u[8] >= -9
 visiting_order_constr[8,7]: - 10 x[8,7] + u[7] - u[8] >= -9
 visiting_order_constr[8,8]: - 10 x[8,8] >= -9
 visiting_order_constr[8,9]: - 10 x[8,9] - u[8] + u[9] >= -9
 visiting_order_constr[8,10]: - 10 x[8,10] - u[8] + u[10] >= -9
 visiting_order_constr[9,1]: - 10 x[9,1] + u[1] - u[9] >= -9
 visiting_order_constr[9,2]: - 10 x[9,2] + u[2] - u[9] >= -9
 visiting_order_constr[9,3]: - 10 x[9,3] + u[3] - u[9] >= -9
 visiting_order_constr[9,4]: - 10 x[9,4] + u[4] - u[9] >= -9
 visiting_order_constr[9,5]: - 10 x[9,5] + u[5] - u[9] >= -9
 visiting_order_constr[9,6]: - 10 x[9,6] + u[6] - u[9] >= -9
 visiting_order_constr[9,7]: - 10 x[9,7] + u[7] - u[9] >= -9
 visiting_order_constr[9,8]: - 10 x[9,8] + u[8] - u[9] >= -9
 visiting_order_constr[9,9]: - 10 x[9,9] >= -9
 visiting_order_constr[9,10]: - 10 x[9,10] - u[9] + u[10] >= -9
 visiting_order_constr[10,1]: - 10 x[10,1] + u[1] - u[10] >= -9
 visiting_order_constr[10,2]: - 10 x[10,2] + u[2] - u[10] >= -9
 visiting_order_constr[10,3]: - 10 x[10,3] + u[3] - u[10] >= -9
 visiting_order_constr[10,4]: - 10 x[10,4] + u[4] - u[10] >= -9
 visiting_order_constr[10,5]: - 10 x[10,5] + u[5] - u[10] >= -9
 visiting_order_constr[10,6]: - 10 x[10,6] + u[6] - u[10] >= -9
 visiting_order_constr[10,7]: - 10 x[10,7] + u[7] - u[10] >= -9
 visiting_order_constr[10,8]: - 10 x[10,8] + u[8] - u[10] >= -9
 visiting_order_constr[10,9]: - 10 x[10,9] + u[9] - u[10] >= -9
 visiting_order_constr[10,10]: - 10 x[10,10] >= -9
 pickup_before_delivery_constr[1]: - u[1] + u[6] >= 1
 pickup_before_delivery_constr[2]: - u[2] + u[7] >= 1
 pickup_before_delivery_constr[3]: - u[3] + u[8] >= 1
 pickup_before_delivery_constr[4]: - u[4] + u[9] >= 1
 pickup_before_delivery_constr[5]: - u[5] + u[10] >= 1
 stack_after_pickup_constr[0,1,0]: - 800 x[0,1] - 400 y[1,0] - s[0,0]
   + s[1,0] >= -800
 stack_after_pickup_constr[0,1,1]: - 800 x[0,1] - 400 y[1,1] - s[0,1]
   + s[1,1] >= -800
 stack_after_pickup_constr[0,1,2]: - 800 x[0,1] - 400 y[1,2] - s[0,2]
   + s[1,2] >= -800
 stack_after_pickup_constr[0,2,0]: - 800 x[0,2] - 400 y[2,0] - s[0,0]
   + s[2,0] >= -800
 stack_after_pickup_constr[0,2,1]: - 800 x[0,2] - 400 y[2,1] - s[0,1]
   + s[2,1] >= -800
 stack_after_pickup_constr[0,2,2]: - 800 x[0,2] - 400 y[2,2] - s[0,2]
   + s[2,2] >= -800
 stack_after_pickup_constr[0,3,0]: - 800 x[0,3] - 400 y[3,0] - s[0,0]
   + s[3,0] >= -800
 stack_after_pickup_constr[0,3,1]: - 800 x[0,3] - 400 y[3,1] - s[0,1]
   + s[3,1] >= -800
 stack_after_pickup_constr[0,3,2]: - 800 x[0,3] - 400 y[3,2] - s[0,2]
   + s[3,2] >= -800
 stack_after_pickup_constr[0,4,0]: - 800 x[0,4] - 400 y[4,0] - s[0,0]
   + s[4,0] >= -800
 stack_after_pickup_constr[0,4,1]: - 800 x[0,4] - 400 y[4,1] - s[0,1]
   + s[4,1] >= -800
 stack_after_pickup_constr[0,4,2]: - 800 x[0,4] - 400 y[4,2] - s[0,2]
   + s[4,2] >= -800
 stack_after_pickup_constr[0,5,0]: - 800 x[0,5] - 400 y[5,0] - s[0,0]
   + s[5,0] >= -800
 stack_after_pickup_constr[0,5,1]: - 800 x[0,5] - 400 y[5,1] - s[0,1]
   + s[5,1] >= -800
 stack_after_pickup_constr[0,5,2]: - 800 x[0,5] - 400 y[5,2] - s[0,2]
   + s[5,2] >= -800
 stack_after_pickup_constr[1,1,0]: - 800 x[1,1] - 400 y[1,0] >= -800
 stack_after_pickup_constr[1,1,1]: - 800 x[1,1] - 400 y[1,1] >= -800
 stack_after_pickup_constr[1,1,2]: - 800 x[1,1] - 400 y[1,2] >= -800
 stack_after_pickup_constr[1,2,0]: - 800 x[1,2] - 400 y[2,0] - s[1,0]
   + s[2,0] >= -800
 stack_after_pickup_constr[1,2,1]: - 800 x[1,2] - 400 y[2,1] - s[1,1]
   + s[2,1] >= -800
 stack_after_pickup_constr[1,2,2]: - 800 x[1,2] - 400 y[2,2] - s[1,2]
   + s[2,2] >= -800
 stack_after_pickup_constr[1,3,0]: - 800 x[1,3] - 400 y[3,0] - s[1,0]
   + s[3,0] >= -800
 stack_after_pickup_constr[1,3,1]: - 800 x[1,3] - 400 y[3,1] - s[1,1]
   + s[3,1] >= -800
 stack_after_pickup_constr[1,3,2]: - 800 x[1,3] - 400 y[3,2] - s[1,2]
   + s[3,2] >= -800
 stack_after_pickup_constr[1,4,0]: - 800 x[1,4] - 400 y[4,0] - s[1,0]
   + s[4,0] >= -800
 stack_after_pickup_constr[1,4,1]: - 800 x[1,4] - 400 y[4,1] - s[1,1]
   + s[4,1] >= -800
 stack_after_pickup_constr[1,4,2]: - 800 x[1,4] - 400 y[4,2] - s[1,2]
   + s[4,2] >= -800
 stack_after_pickup_constr[1,5,0]: - 800 x[1,5] - 400 y[5,0] - s[1,0]
   + s[5,0] >= -800
 stack_after_pickup_constr[1,5,1]: - 800 x[1,5] - 400 y[5,1] - s[1,1]
   + s[5,1] >= -800
 stack_after_pickup_constr[1,5,2]: - 800 x[1,5] - 400 y[5,2] - s[1,2]
   + s[5,2] >= -800
 stack_after_pickup_constr[2,1,0]: - 800 x[2,1] - 400 y[1,0] + s[1,0]
   - s[2,0] >= -800
 stack_after_pickup_constr[2,1,1]: - 800 x[2,1] - 400 y[1,1] + s[1,1]
   - s[2,1] >= -800
 stack_after_pickup_constr[2,1,2]: - 800 x[2,1] - 400 y[1,2] + s[1,2]
   - s[2,2] >= -800
 stack_after_pickup_constr[2,2,0]: - 800 x[2,2] - 400 y[2,0] >= -800
 stack_after_pickup_constr[2,2,1]: - 800 x[2,2] - 400 y[2,1] >= -800
 stack_after_pickup_constr[2,2,2]: - 800 x[2,2] - 400 y[2,2] >= -800
 stack_after_pickup_constr[2,3,0]: - 800 x[2,3] - 400 y[3,0] - s[2,0]
   + s[3,0] >= -800
 stack_after_pickup_constr[2,3,1]: - 800 x[2,3] - 400 y[3,1] - s[2,1]
   + s[3,1] >= -800
 stack_after_pickup_constr[2,3,2]: - 800 x[2,3] - 400 y[3,2] - s[2,2]
   + s[3,2] >= -800
 stack_after_pickup_constr[2,4,0]: - 800 x[2,4] - 400 y[4,0] - s[2,0]
   + s[4,0] >= -800
 stack_after_pickup_constr[2,4,1]: - 800 x[2,4] - 400 y[4,1] - s[2,1]
   + s[4,1] >= -800
 stack_after_pickup_constr[2,4,2]: - 800 x[2,4] - 400 y[4,2] - s[2,2]
   + s[4,2] >= -800
 stack_after_pickup_constr[2,5,0]: - 800 x[2,5] - 400 y[5,0] - s[2,0]
   + s[5,0] >= -800
 stack_after_pickup_constr[2,5,1]: - 800 x[2,5] - 400 y[5,1] - s[2,1]
   + s[5,1] >= -800
 stack_after_pickup_constr[2,5,2]: - 800 x[2,5] - 400 y[5,2] - s[2,2]
   + s[5,2] >= -800
 stack_after_pickup_constr[3,1,0]: - 800 x[3,1] - 400 y[1,0] + s[1,0]
   - s[3,0] >= -800
 stack_after_pickup_constr[3,1,1]: - 800 x[3,1] - 400 y[1,1] + s[1,1]
   - s[3,1] >= -800
 stack_after_pickup_constr[3,1,2]: - 800 x[3,1] - 400 y[1,2] + s[1,2]
   - s[3,2] >= -800
 stack_after_pickup_constr[3,2,0]: - 800 x[3,2] - 400 y[2,0] + s[2,0]
   - s[3,0] >= -800
 stack_after_pickup_constr[3,2,1]: - 800 x[3,2] - 400 y[2,1] + s[2,1]
   - s[3,1] >= -800
 stack_after_pickup_constr[3,2,2]: - 800 x[3,2] - 400 y[2,2] + s[2,2]
   - s[3,2] >= -800
 stack_after_pickup_constr[3,3,0]: - 800 x[3,3] - 400 y[3,0] >= -800
 stack_after_pickup_constr[3,3,1]: - 800 x[3,3] - 400 y[3,1] >= -800
 stack_after_pickup_constr[3,3,2]: - 800 x[3,3] - 400 y[3,2] >= -800
 stack_after_pickup_constr[3,4,0]: - 800 x[3,4] - 400 y[4,0] - s[3,0]
   + s[4,0] >= -800
 stack_after_pickup_constr[3,4,1]: - 800 x[3,4] - 400 y[4,1] - s[3,1]
   + s[4,1] >= -800
 stack_after_pickup_constr[3,4,2]: - 800 x[3,4] - 400 y[4,2] - s[3,2]
   + s[4,2] >= -800
 stack_after_pickup_constr[3,5,0]: - 800 x[3,5] - 400 y[5,0] - s[3,0]
   + s[5,0] >= -800
 stack_after_pickup_constr[3,5,1]: - 800 x[3,5] - 400 y[5,1] - s[3,1]
   + s[5,1] >= -800
 stack_after_pickup_constr[3,5,2]: - 800 x[3,5] - 400 y[5,2] - s[3,2]
   + s[5,2] >= -800
 stack_after_pickup_constr[4,1,0]: - 800 x[4,1] - 400 y[1,0] + s[1,0]
   - s[4,0] >= -800
 stack_after_pickup_constr[4,1,1]: - 800 x[4,1] - 400 y[1,1] + s[1,1]
   - s[4,1] >= -800
 stack_after_pickup_constr[4,1,2]: - 800 x[4,1] - 400 y[1,2] + s[1,2]
   - s[4,2] >= -800
 stack_after_pickup_constr[4,2,0]: - 800 x[4,2] - 400 y[2,0] + s[2,0]
   - s[4,0] >= -800
 stack_after_pickup_constr[4,2,1]: - 800 x[4,2] - 400 y[2,1] + s[2,1]
   - s[4,1] >= -800
 stack_after_pickup_constr[4,2,2]: - 800 x[4,2] - 400 y[2,2] + s[2,2]
   - s[4,2] >= -800
 stack_after_pickup_constr[4,3,0]: - 800 x[4,3] - 400 y[3,0] + s[3,0]
   - s[4,0] >= -800
 stack_after_pickup_constr[4,3,1]: - 800 x[4,3] - 400 y[3,1] + s[3,1]
   - s[4,1] >= -800
 stack_after_pickup_constr[4,3,2]: - 800 x[4,3] - 400 y[3,2] + s[3,2]
   - s[4,2] >= -800
 stack_after_pickup_constr[4,4,0]: - 800 x[4,4] - 400 y[4,0] >= -800
 stack_after_pickup_constr[4,4,1]: - 800 x[4,4] - 400 y[4,1] >= -800
 stack_after_pickup_constr[4,4,2]: - 800 x[4,4] - 400 y[4,2] >= -800
 stack_after_pickup_constr[4,5,0]: - 800 x[4,5] - 400 y[5,0] - s[4,0]
   + s[5,0] >= -800
 stack_after_pickup_constr[4,5,1]: - 800 x[4,5] - 400 y[5,1] - s[4,1]
   + s[5,1] >= -800
 stack_after_pickup_constr[4,5,2]: - 800 x[4,5] - 400 y[5,2] - s[4,2]
   + s[5,2] >= -800
 stack_after_pickup_constr[5,1,0]: - 800 x[5,1] - 400 y[1,0] + s[1,0]
   - s[5,0] >= -800
 stack_after_pickup_constr[5,1,1]: - 800 x[5,1] - 400 y[1,1] + s[1,1]
   - s[5,1] >= -800
 stack_after_pickup_constr[5,1,2]: - 800 x[5,1] - 400 y[1,2] + s[1,2]
   - s[5,2] >= -800
 stack_after_pickup_constr[5,2,0]: - 800 x[5,2] - 400 y[2,0] + s[2,0]
   - s[5,0] >= -800
 stack_after_pickup_constr[5,2,1]: - 800 x[5,2] - 400 y[2,1] + s[2,1]
   - s[5,1] >= -800
 stack_after_pickup_constr[5,2,2]: - 800 x[5,2] - 400 y[2,2] + s[2,2]
   - s[5,2] >= -800
 stack_after_pickup_constr[5,3,0]: - 800 x[5,3] - 400 y[3,0] + s[3,0]
   - s[5,0] >= -800
 stack_after_pickup_constr[5,3,1]: - 800 x[5,3] - 400 y[3,1] + s[3,1]
   - s[5,1] >= -800
 stack_after_pickup_constr[5,3,2]: - 800 x[5,3] - 400 y[3,2] + s[3,2]
   - s[5,2] >= -800
 stack_after_pickup_constr[5,4,0]: - 800 x[5,4] - 400 y[4,0] + s[4,0]
   - s[5,0] >= -800
 stack_after_pickup_constr[5,4,1]: - 800 x[5,4] - 400 y[4,1] + s[4,1]
   - s[5,1] >= -800
 stack_after_pickup_constr[5,4,2]: - 800 x[5,4] - 400 y[4,2] + s[4,2]
   - s[5,2] >= -800
 stack_after_pickup_constr[5,5,0]: - 800 x[5,5] - 400 y[5,0] >= -800
 stack_after_pickup_constr[5,5,1]: - 800 x[5,5] - 400 y[5,1] >= -800
 stack_after_pickup_constr[5,5,2]: - 800 x[5,5] - 400 y[5,2] >= -800
 stack_after_pickup_constr[6,1,0]: - 800 x[6,1] - 400 y[1,0] + s[1,0]
   - s[6,0] >= -800
 stack_after_pickup_constr[6,1,1]: - 800 x[6,1] - 400 y[1,1] + s[1,1]
   - s[6,1] >= -800
 stack_after_pickup_constr[6,1,2]: - 800 x[6,1] - 400 y[1,2] + s[1,2]
   - s[6,2] >= -800
 stack_after_pickup_constr[6,2,0]: - 800 x[6,2] - 400 y[2,0] + s[2,0]
   - s[6,0] >= -800
 stack_after_pickup_constr[6,2,1]: - 800 x[6,2] - 400 y[2,1] + s[2,1]
   - s[6,1] >= -800
 stack_after_pickup_constr[6,2,2]: - 800 x[6,2] - 400 y[2,2] + s[2,2]
   - s[6,2] >= -800
 stack_after_pickup_constr[6,3,0]: - 800 x[6,3] - 400 y[3,0] + s[3,0]
   - s[6,0] >= -800
 stack_after_pickup_constr[6,3,1]: - 800 x[6,3] - 400 y[3,1] + s[3,1]
   - s[6,1] >= -800
 stack_after_pickup_constr[6,3,2]: - 800 x[6,3] - 400 y[3,2] + s[3,2]
   - s[6,2] >= -800
 stack_after_pickup_constr[6,4,0]: - 800 x[6,4] - 400 y[4,0] + s[4,0]
   - s[6,0] >= -800
 stack_after_pickup_constr[6,4,1]: - 800 x[6,4] - 400 y[4,1] + s[4,1]
   - s[6,1] >= -800
 stack_after_pickup_constr[6,4,2]: - 800 x[6,4] - 400 y[4,2] + s[4,2]
   - s[6,2] >= -800
 stack_after_pickup_constr[6,5,0]: - 800 x[6,5] - 400 y[5,0] + s[5,0]
   - s[6,0] >= -800
 stack_after_pickup_constr[6,5,1]: - 800 x[6,5] - 400 y[5,1] + s[5,1]
   - s[6,1] >= -800
 stack_after_pickup_constr[6,5,2]: - 800 x[6,5] - 400 y[5,2] + s[5,2]
   - s[6,2] >= -800
 stack_after_pickup_constr[7,1,0]: - 800 x[7,1] - 400 y[1,0] + s[1,0]
   - s[7,0] >= -800
 stack_after_pickup_constr[7,1,1]: - 800 x[7,1] - 400 y[1,1] + s[1,1]
   - s[7,1] >= -800
 stack_after_pickup_constr[7,1,2]: - 800 x[7,1] - 400 y[1,2] + s[1,2]
   - s[7,2] >= -800
 stack_after_pickup_constr[7,2,0]: - 800 x[7,2] - 400 y[2,0] + s[2,0]
   - s[7,0] >= -800
 stack_after_pickup_constr[7,2,1]: - 800 x[7,2] - 400 y[2,1] + s[2,1]
   - s[7,1] >= -800
 stack_after_pickup_constr[7,2,2]: - 800 x[7,2] - 400 y[2,2] + s[2,2]
   - s[7,2] >= -800
 stack_after_pickup_constr[7,3,0]: - 800 x[7,3] - 400 y[3,0] + s[3,0]
   - s[7,0] >= -800
 stack_after_pickup_constr[7,3,1]: - 800 x[7,3] - 400 y[3,1] + s[3,1]
   - s[7,1] >= -800
 stack_after_pickup_constr[7,3,2]: - 800 x[7,3] - 400 y[3,2] + s[3,2]
   - s[7,2] >= -800
 stack_after_pickup_constr[7,4,0]: - 800 x[7,4] - 400 y[4,0] + s[4,0]
   - s[7,0] >= -800
 stack_after_pickup_constr[7,4,1]: - 800 x[7,4] - 400 y[4,1] + s[4,1]
   - s[7,1] >= -800
 stack_after_pickup_constr[7,4,2]: - 800 x[7,4] - 400 y[4,2] + s[4,2]
   - s[7,2] >= -800
 stack_after_pickup_constr[7,5,0]: - 800 x[7,5] - 400 y[5,0] + s[5,0]
   - s[7,0] >= -800
 stack_after_pickup_constr[7,5,1]: - 800 x[7,5] - 400 y[5,1] + s[5,1]
   - s[7,1] >= -800
 stack_after_pickup_constr[7,5,2]: - 800 x[7,5] - 400 y[5,2] + s[5,2]
   - s[7,2] >= -800
 stack_after_pickup_constr[8,1,0]: - 800 x[8,1] - 400 y[1,0] + s[1,0]
   - s[8,0] >= -800
 stack_after_pickup_constr[8,1,1]: - 800 x[8,1] - 400 y[1,1] + s[1,1]
   - s[8,1] >= -800
 stack_after_pickup_constr[8,1,2]: - 800 x[8,1] - 400 y[1,2] + s[1,2]
   - s[8,2] >= -800
 stack_after_pickup_constr[8,2,0]: - 800 x[8,2] - 400 y[2,0] + s[2,0]
   - s[8,0] >= -800
 stack_after_pickup_constr[8,2,1]: - 800 x[8,2] - 400 y[2,1] + s[2,1]
   - s[8,1] >= -800
 stack_after_pickup_constr[8,2,2]: - 800 x[8,2] - 400 y[2,2] + s[2,2]
   - s[8,2] >= -800
 stack_after_pickup_constr[8,3,0]: - 800 x[8,3] - 400 y[3,0] + s[3,0]
   - s[8,0] >= -800
 stack_after_pickup_constr[8,3,1]: - 800 x[8,3] - 400 y[3,1] + s[3,1]
   - s[8,1] >= -800
 stack_after_pickup_constr[8,3,2]: - 800 x[8,3] - 400 y[3,2] + s[3,2]
   - s[8,2] >= -800
 stack_after_pickup_constr[8,4,0]: - 800 x[8,4] - 400 y[4,0] + s[4,0]
   - s[8,0] >= -800
 stack_after_pickup_constr[8,4,1]: - 800 x[8,4] - 400 y[4,1] + s[4,1]
   - s[8,1] >= -800
 stack_after_pickup_constr[8,4,2]: - 800 x[8,4] - 400 y[4,2] + s[4,2]
   - s[8,2] >= -800
 stack_after_pickup_constr[8,5,0]: - 800 x[8,5] - 400 y[5,0] + s[5,0]
   - s[8,0] >= -800
 stack_after_pickup_constr[8,5,1]: - 800 x[8,5] - 400 y[5,1] + s[5,1]
   - s[8,1] >= -800
 stack_after_pickup_constr[8,5,2]: - 800 x[8,5] - 400 y[5,2] + s[5,2]
   - s[8,2] >= -800
 stack_after_pickup_constr[9,1,0]: - 800 x[9,1] - 400 y[1,0] + s[1,0]
   - s[9,0] >= -800
 stack_after_pickup_constr[9,1,1]: - 800 x[9,1] - 400 y[1,1] + s[1,1]
   - s[9,1] >= -800
 stack_after_pickup_constr[9,1,2]: - 800 x[9,1] - 400 y[1,2] + s[1,2]
   - s[9,2] >= -800
 stack_after_pickup_constr[9,2,0]: - 800 x[9,2] - 400 y[2,0] + s[2,0]
   - s[9,0] >= -800
 stack_after_pickup_constr[9,2,1]: - 800 x[9,2] - 400 y[2,1] + s[2,1]
   - s[9,1] >= -800
 stack_after_pickup_constr[9,2,2]: - 800 x[9,2] - 400 y[2,2] + s[2,2]
   - s[9,2] >= -800
 stack_after_pickup_constr[9,3,0]: - 800 x[9,3] - 400 y[3,0] + s[3,0]
   - s[9,0] >= -800
 stack_after_pickup_constr[9,3,1]: - 800 x[9,3] - 400 y[3,1] + s[3,1]
   - s[9,1] >= -800
 stack_after_pickup_constr[9,3,2]: - 800 x[9,3] - 400 y[3,2] + s[3,2]
   - s[9,2] >= -800
 stack_after_pickup_constr[9,4,0]: - 800 x[9,4] - 400 y[4,0] + s[4,0]
   - s[9,0] >= -800
 stack_after_pickup_constr[9,4,1]: - 800 x[9,4] - 400 y[4,1] + s[4,1]
   - s[9,1] >= -800
 stack_after_pickup_constr[9,4,2]: - 800 x[9,4] - 400 y[4,2] + s[4,2]
   - s[9,2] >= -800
 stack_after_pickup_constr[9,5,0]: - 800 x[9,5] - 400 y[5,0] + s[5,0]
   - s[9,0] >= -800
 stack_after_pickup_constr[9,5,1]: - 800 x[9,5] - 400 y[5,1] + s[5,1]
   - s[9,1] >= -800
 stack_after_pickup_constr[9,5,2]: - 800 x[9,5] - 400 y[5,2] + s[5,2]
   - s[9,2] >= -800
 stack_after_pickup_constr[10,1,0]: - 800 x[10,1] - 400 y[1,0] + s[1,0]
   - s[10,0] >= -800
 stack_after_pickup_constr[10,1,1]: - 800 x[10,1] - 400 y[1,1] + s[1,1]
   - s[10,1] >= -800
 stack_after_pickup_constr[10,1,2]: - 800 x[10,1] - 400 y[1,2] + s[1,2]
   - s[10,2] >= -800
 stack_after_pickup_constr[10,2,0]: - 800 x[10,2] - 400 y[2,0] + s[2,0]
   - s[10,0] >= -800
 stack_after_pickup_constr[10,2,1]: - 800 x[10,2] - 400 y[2,1] + s[2,1]
   - s[10,1] >= -800
 stack_after_pickup_constr[10,2,2]: - 800 x[10,2] - 400 y[2,2] + s[2,2]
   - s[10,2] >= -800
 stack_after_pickup_constr[10,3,0]: - 800 x[10,3] - 400 y[3,0] + s[3,0]
   - s[10,0] >= -800
 stack_after_pickup_constr[10,3,1]: - 800 x[10,3] - 400 y[3,1] + s[3,1]
   - s[10,1] >= -800
 stack_after_pickup_constr[10,3,2]: - 800 x[10,3] - 400 y[3,2] + s[3,2]
   - s[10,2] >= -800
 stack_after_pickup_constr[10,4,0]: - 800 x[10,4] - 400 y[4,0] + s[4,0]
   - s[10,0] >= -800
 stack_after_pickup_constr[10,4,1]: - 800 x[10,4] - 400 y[4,1] + s[4,1]
   - s[10,1] >= -800
 stack_after_pickup_constr[10,4,2]: - 800 x[10,4] - 400 y[4,2] + s[4,2]
   - s[10,2] >= -800
 stack_after_pickup_constr[10,5,0]: - 800 x[10,5] - 400 y[5,0] + s[5,0]
   - s[10,0] >= -800
 stack_after_pickup_constr[10,5,1]: - 800 x[10,5] - 400 y[5,1] + s[5,1]
   - s[10,1] >= -800
 stack_after_pickup_constr[10,5,2]: - 800 x[10,5] - 400 y[5,2] + s[5,2]
   - s[10,2] >= -800
 stack_after_delivery_constr[0,1,0]: - 800 x[0,6] + 400 y[1,0] - s[0,0]
   + s[6,0] >= -800
 stack_after_delivery_constr[0,1,1]: - 800 x[0,6] + 400 y[1,1] - s[0,1]
   + s[6,1] >= -800
 stack_after_delivery_constr[0,1,2]: - 800 x[0,6] + 400 y[1,2] - s[0,2]
   + s[6,2] >= -800
 stack_after_delivery_constr[0,2,0]: - 800 x[0,7] + 400 y[2,0] - s[0,0]
   + s[7,0] >= -800
 stack_after_delivery_constr[0,2,1]: - 800 x[0,7] + 400 y[2,1] - s[0,1]
   + s[7,1] >= -800
 stack_after_delivery_constr[0,2,2]: - 800 x[0,7] + 400 y[2,2] - s[0,2]
   + s[7,2] >= -800
 stack_after_delivery_constr[0,3,0]: - 800 x[0,8] + 400 y[3,0] - s[0,0]
   + s[8,0] >= -800
 stack_after_delivery_constr[0,3,1]: - 800 x[0,8] + 400 y[3,1] - s[0,1]
   + s[8,1] >= -800
 stack_after_delivery_constr[0,3,2]: - 800 x[0,8] + 400 y[3,2] - s[0,2]
   + s[8,2] >= -800
 stack_after_delivery_constr[0,4,0]: - 800 x[0,9] + 400 y[4,0] - s[0,0]
   + s[9,0] >= -800
 stack_after_delivery_constr[0,4,1]: - 800 x[0,9] + 400 y[4,1] - s[0,1]
   + s[9,1] >= -800
 stack_after_delivery_constr[0,4,2]: - 800 x[0,9] + 400 y[4,2] - s[0,2]
   + s[9,2] >= -800
 stack_after_delivery_constr[0,5,0]: - 800 x[0,10] + 400 y[5,0] - s[0,0]
   + s[10,0] >= -800
 stack_after_delivery_constr[0,5,1]: - 800 x[0,10] + 400 y[5,1] - s[0,1]
   + s[10,1] >= -800
 stack_after_delivery_constr[0,5,2]: - 800 x[0,10] + 400 y[5,2] - s[0,2]
   + s[10,2] >= -800
 stack_after_delivery_constr[1,1,0]: - 800 x[1,6] + 400 y[1,0] - s[1,0]
   + s[6,0] >= -800
 stack_after_delivery_constr[1,1,1]: - 800 x[1,6] + 400 y[1,1] - s[1,1]
   + s[6,1] >= -800
 stack_after_delivery_constr[1,1,2]: - 800 x[1,6] + 400 y[1,2] - s[1,2]
   + s[6,2] >= -800
 stack_after_delivery_constr[1,2,0]: - 800 x[1,7] + 400 y[2,0] - s[1,0]
   + s[7,0] >= -800
 stack_after_delivery_constr[1,2,1]: - 800 x[1,7] + 400 y[2,1] - s[1,1]
   + s[7,1] >= -800
 stack_after_delivery_constr[1,2,2]: - 800 x[1,7] + 400 y[2,2] - s[1,2]
   + s[7,2] >= -800
 stack_after_delivery_constr[1,3,0]: - 800 x[1,8] + 400 y[3,0] - s[1,0]
   + s[8,0] >= -800
 stack_after_delivery_constr[1,3,1]: - 800 x[1,8] + 400 y[3,1] - s[1,1]
   + s[8,1] >= -800
 stack_after_delivery_constr[1,3,2]: - 800 x[1,8] + 400 y[3,2] - s[1,2]
   + s[8,2] >= -800
 stack_after_delivery_constr[1,4,0]: - 800 x[1,9] + 400 y[4,0] - s[1,0]
   + s[9,0] >= -800
 stack_after_delivery_constr[1,4,1]: - 800 x[1,9] + 400 y[4,1] - s[1,1]
   + s[9,1] >= -800
 stack_after_delivery_constr[1,4,2]: - 800 x[1,9] + 400 y[4,2] - s[1,2]
   + s[9,2] >= -800
 stack_after_delivery_constr[1,5,0]: - 800 x[1,10] + 400 y[5,0] - s[1,0]
   + s[10,0] >= -800
 stack_after_delivery_constr[1,5,1]: - 800 x[1,10] + 400 y[5,1] - s[1,1]
   + s[10,1] >= -800
 stack_after_delivery_constr[1,5,2]: - 800 x[1,10] + 400 y[5,2] - s[1,2]
   + s[10,2] >= -800
 stack_after_delivery_constr[2,1,0]: - 800 x[2,6] + 400 y[1,0] - s[2,0]
   + s[6,0] >= -800
 stack_after_delivery_constr[2,1,1]: - 800 x[2,6] + 400 y[1,1] - s[2,1]
   + s[6,1] >= -800
 stack_after_delivery_constr[2,1,2]: - 800 x[2,6] + 400 y[1,2] - s[2,2]
   + s[6,2] >= -800
 stack_after_delivery_constr[2,2,0]: - 800 x[2,7] + 400 y[2,0] - s[2,0]
   + s[7,0] >= -800
 stack_after_delivery_constr[2,2,1]: - 800 x[2,7] + 400 y[2,1] - s[2,1]
   + s[7,1] >= -800
 stack_after_delivery_constr[2,2,2]: - 800 x[2,7] + 400 y[2,2] - s[2,2]
   + s[7,2] >= -800
 stack_after_delivery_constr[2,3,0]: - 800 x[2,8] + 400 y[3,0] - s[2,0]
   + s[8,0] >= -800
 stack_after_delivery_constr[2,3,1]: - 800 x[2,8] + 400 y[3,1] - s[2,1]
   + s[8,1] >= -800
 stack_after_delivery_constr[2,3,2]: - 800 x[2,8] + 400 y[3,2] - s[2,2]
   + s[8,2] >= -800
 stack_after_delivery_constr[2,4,0]: - 800 x[2,9] + 400 y[4,0] - s[2,0]
   + s[9,0] >= -800
 stack_after_delivery_constr[2,4,1]: - 800 x[2,9] + 400 y[4,1] - s[2,1]
   + s[9,1] >= -800
 stack_after_delivery_constr[2,4,2]: - 800 x[2,9] + 400 y[4,2] - s[2,2]
   + s[9,2] >= -800
 stack_after_delivery_constr[2,5,0]: - 800 x[2,10] + 400 y[5,0] - s[2,0]
   + s[10,0] >= -800
 stack_after_delivery_constr[2,5,1]: - 800 x[2,10] + 400 y[5,1] - s[2,1]
   + s[10,1] >= -800
 stack_after_delivery_constr[2,5,2]: - 800 x[2,10] + 400 y[5,2] - s[2,2]
   + s[10,2] >= -800
 stack_after_delivery_constr[3,1,0]: - 800 x[3,6] + 400 y[1,0] - s[3,0]
   + s[6,0] >= -800
 stack_after_delivery_constr[3,1,1]: - 800 x[3,6] + 400 y[1,1] - s[3,1]
   + s[6,1] >= -800
 stack_after_delivery_constr[3,1,2]: - 800 x[3,6] + 400 y[1,2] - s[3,2]
   + s[6,2] >= -800
 stack_after_delivery_constr[3,2,0]: - 800 x[3,7] + 400 y[2,0] - s[3,0]
   + s[7,0] >= -800
 stack_after_delivery_constr[3,2,1]: - 800 x[3,7] + 400 y[2,1] - s[3,1]
   + s[7,1] >= -800
 stack_after_delivery_constr[3,2,2]: - 800 x[3,7] + 400 y[2,2] - s[3,2]
   + s[7,2] >= -800
 stack_after_delivery_constr[3,3,0]: - 800 x[3,8] + 400 y[3,0] - s[3,0]
   + s[8,0] >= -800
 stack_after_delivery_constr[3,3,1]: - 800 x[3,8] + 400 y[3,1] - s[3,1]
   + s[8,1] >= -800
 stack_after_delivery_constr[3,3,2]: - 800 x[3,8] + 400 y[3,2] - s[3,2]
   + s[8,2] >= -800
 stack_after_delivery_constr[3,4,0]: - 800 x[3,9] + 400 y[4,0] - s[3,0]
   + s[9,0] >= -800
 stack_after_delivery_constr[3,4,1]: - 800 x[3,9] + 400 y[4,1] - s[3,1]
   + s[9,1] >= -800
 stack_after_delivery_constr[3,4,2]: - 800 x[3,9] + 400 y[4,2] - s[3,2]
   + s[9,2] >= -800
 stack_after_delivery_constr[3,5,0]: - 800 x[3,10] + 400 y[5,0] - s[3,0]
   + s[10,0] >= -800
 stack_after_delivery_constr[3,5,1]: - 800 x[3,10] + 400 y[5,1] - s[3,1]
   + s[10,1] >= -800
 stack_after_delivery_constr[3,5,2]: - 800 x[3,10] + 400 y[5,2] - s[3,2]
   + s[10,2] >= -800
 stack_after_delivery_constr[4,1,0]: - 800 x[4,6] + 400 y[1,0] - s[4,0]
   + s[6,0] >= -800
 stack_after_delivery_constr[4,1,1]: - 800 x[4,6] + 400 y[1,1] - s[4,1]
   + s[6,1] >= -800
 stack_after_delivery_constr[4,1,2]: - 800 x[4,6] + 400 y[1,2] - s[4,2]
   + s[6,2] >= -800
 stack_after_delivery_constr[4,2,0]: - 800 x[4,7] + 400 y[2,0] - s[4,0]
   + s[7,0] >= -800
 stack_after_delivery_constr[4,2,1]: - 800 x[4,7] + 400 y[2,1] - s[4,1]
   + s[7,1] >= -800
 stack_after_delivery_constr[4,2,2]: - 800 x[4,7] + 400 y[2,2] - s[4,2]
   + s[7,2] >= -800
 stack_after_delivery_constr[4,3,0]: - 800 x[4,8] + 400 y[3,0] - s[4,0]
   + s[8,0] >= -800
 stack_after_delivery_constr[4,3,1]: - 800 x[4,8] + 400 y[3,1] - s[4,1]
   + s[8,1] >= -800
 stack_after_delivery_constr[4,3,2]: - 800 x[4,8] + 400 y[3,2] - s[4,2]
   + s[8,2] >= -800
 stack_after_delivery_constr[4,4,0]: - 800 x[4,9] + 400 y[4,0] - s[4,0]
   + s[9,0] >= -800
 stack_after_delivery_constr[4,4,1]: - 800 x[4,9] + 400 y[4,1] - s[4,1]
   + s[9,1] >= -800
 stack_after_delivery_constr[4,4,2]: - 800 x[4,9] + 400 y[4,2] - s[4,2]
   + s[9,2] >= -800
 stack_after_delivery_constr[4,5,0]: - 800 x[4,10] + 400 y[5,0] - s[4,0]
   + s[10,0] >= -800
 stack_after_delivery_constr[4,5,1]: - 800 x[4,10] + 400 y[5,1] - s[4,1]
   + s[10,1] >= -800
 stack_after_delivery_constr[4,5,2]: - 800 x[4,10] + 400 y[5,2] - s[4,2]
   + s[10,2] >= -800
 stack_after_delivery_constr[5,1,0]: - 800 x[5,6] + 400 y[1,0] - s[5,0]
   + s[6,0] >= -800
 stack_after_delivery_constr[5,1,1]: - 800 x[5,6] + 400 y[1,1] - s[5,1]
   + s[6,1] >= -800
 stack_after_delivery_constr[5,1,2]: - 800 x[5,6] + 400 y[1,2] - s[5,2]
   + s[6,2] >= -800
 stack_after_delivery_constr[5,2,0]: - 800 x[5,7] + 400 y[2,0] - s[5,0]
   + s[7,0] >= -800
 stack_after_delivery_constr[5,2,1]: - 800 x[5,7] + 400 y[2,1] - s[5,1]
   + s[7,1] >= -800
 stack_after_delivery_constr[5,2,2]: - 800 x[5,7] + 400 y[2,2] - s[5,2]
   + s[7,2] >= -800
 stack_after_delivery_constr[5,3,0]: - 800 x[5,8] + 400 y[3,0] - s[5,0]
   + s[8,0] >= -800
 stack_after_delivery_constr[5,3,1]: - 800 x[5,8] + 400 y[3,1] - s[5,1]
   + s[8,1] >= -800
 stack_after_delivery_constr[5,3,2]: - 800 x[5,8] + 400 y[3,2] - s[5,2]
   + s[8,2] >= -800
 stack_after_delivery_constr[5,4,0]: - 800 x[5,9] + 400 y[4,0] - s[5,0]
   + s[9,0] >= -800
 stack_after_delivery_constr[5,4,1]: - 800 x[5,9] + 400 y[4,1] - s[5,1]
   + s[9,1] >= -800
 stack_after_delivery_constr[5,4,2]: - 800 x[5,9] + 400 y[4,2] - s[5,2]
   + s[9,2] >= -800
 stack_after_delivery_constr[5,5,0]: - 800 x[5,10] + 400 y[5,0] - s[5,0]
   + s[10,0] >= -800
 stack_after_delivery_constr[5,5,1]: - 800 x[5,10] + 400 y[5,1] - s[5,1]
   + s[10,1] >= -800
 stack_after_delivery_constr[5,5,2]: - 800 x[5,10] + 400 y[5,2] - s[5,2]
   + s[10,2] >= -800
 stack_after_delivery_constr[6,1,0]: - 800 x[6,6] + 400 y[1,0] >= -800
 stack_after_delivery_constr[6,1,1]: - 800 x[6,6] + 400 y[1,1] >= -800
 stack_after_delivery_constr[6,1,2]: - 800 x[6,6] + 400 y[1,2] >= -800
 stack_after_delivery_constr[6,2,0]: - 800 x[6,7] + 400 y[2,0] - s[6,0]
   + s[7,0] >= -800
 stack_after_delivery_constr[6,2,1]: - 800 x[6,7] + 400 y[2,1] - s[6,1]
   + s[7,1] >= -800
 stack_after_delivery_constr[6,2,2]: - 800 x[6,7] + 400 y[2,2] - s[6,2]
   + s[7,2] >= -800
 stack_after_delivery_constr[6,3,0]: - 800 x[6,8] + 400 y[3,0] - s[6,0]
   + s[8,0] >= -800
 stack_after_delivery_constr[6,3,1]: - 800 x[6,8] + 400 y[3,1] - s[6,1]
   + s[8,1] >= -800
 stack_after_delivery_constr[6,3,2]: - 800 x[6,8] + 400 y[3,2] - s[6,2]
   + s[8,2] >= -800
 stack_after_delivery_constr[6,4,0]: - 800 x[6,9] + 400 y[4,0] - s[6,0]
   + s[9,0] >= -800
 stack_after_delivery_constr[6,4,1]: - 800 x[6,9] + 400 y[4,1] - s[6,1]
   + s[9,1] >= -800
 stack_after_delivery_constr[6,4,2]: - 800 x[6,9] + 400 y[4,2] - s[6,2]
   + s[9,2] >= -800
 stack_after_delivery_constr[6,5,0]: - 800 x[6,10] + 400 y[5,0] - s[6,0]
   + s[10,0] >= -800
 stack_after_delivery_constr[6,5,1]: - 800 x[6,10] + 400 y[5,1] - s[6,1]
   + s[10,1] >= -800
 stack_after_delivery_constr[6,5,2]: - 800 x[6,10] + 400 y[5,2] - s[6,2]
   + s[10,2] >= -800
 stack_after_delivery_constr[7,1,0]: - 800 x[7,6] + 400 y[1,0] + s[6,0]
   - s[7,0] >= -800
 stack_after_delivery_constr[7,1,1]: - 800 x[7,6] + 400 y[1,1] + s[6,1]
   - s[7,1] >= -800
 stack_after_delivery_constr[7,1,2]: - 800 x[7,6] + 400 y[1,2] + s[6,2]
   - s[7,2] >= -800
 stack_after_delivery_constr[7,2,0]: - 800 x[7,7] + 400 y[2,0] >= -800
 stack_after_delivery_constr[7,2,1]: - 800 x[7,7] + 400 y[2,1] >= -800
 stack_after_delivery_constr[7,2,2]: - 800 x[7,7] + 400 y[2,2] >= -800
 stack_after_delivery_constr[7,3,0]: - 800 x[7,8] + 400 y[3,0] - s[7,0]
   + s[8,0] >= -800
 stack_after_delivery_constr[7,3,1]: - 800 x[7,8] + 400 y[3,1] - s[7,1]
   + s[8,1] >= -800
 stack_after_delivery_constr[7,3,2]: - 800 x[7,8] + 400 y[3,2] - s[7,2]
   + s[8,2] >= -800
 stack_after_delivery_constr[7,4,0]: - 800 x[7,9] + 400 y[4,0] - s[7,0]
   + s[9,0] >= -800
 stack_after_delivery_constr[7,4,1]: - 800 x[7,9] + 400 y[4,1] - s[7,1]
   + s[9,1] >= -800
 stack_after_delivery_constr[7,4,2]: - 800 x[7,9] + 400 y[4,2] - s[7,2]
   + s[9,2] >= -800
 stack_after_delivery_constr[7,5,0]: - 800 x[7,10] + 400 y[5,0] - s[7,0]
   + s[10,0] >= -800
 stack_after_delivery_constr[7,5,1]: - 800 x[7,10] + 400 y[5,1] - s[7,1]
   + s[10,1] >= -800
 stack_after_delivery_constr[7,5,2]: - 800 x[7,10] + 400 y[5,2] - s[7,2]
   + s[10,2] >= -800
 stack_after_delivery_constr[8,1,0]: - 800 x[8,6] + 400 y[1,0] + s[6,0]
   - s[8,0] >= -800
 stack_after_delivery_constr[8,1,1]: - 800 x[8,6] + 400 y[1,1] + s[6,1]
   - s[8,1] >= -800
 stack_after_delivery_constr[8,1,2]: - 800 x[8,6] + 400 y[1,2] + s[6,2]
   - s[8,2] >= -800
 stack_after_delivery_constr[8,2,0]: - 800 x[8,7] + 400 y[2,0] + s[7,0]
   - s[8,0] >= -800
 stack_after_delivery_constr[8,2,1]: - 800 x[8,7] + 400 y[2,1] + s[7,1]
   - s[8,1] >= -800
 stack_after_delivery_constr[8,2,2]: - 800 x[8,7] + 400 y[2,2] + s[7,2]
   - s[8,2] >= -800
 stack_after_delivery_constr[8,3,0]: - 800 x[8,8] + 400 y[3,0] >= -800
 stack_after_delivery_constr[8,3,1]: - 800 x[8,8] + 400 y[3,1] >= -800
 stack_after_delivery_constr[8,3,2]: - 800 x[8,8] + 400 y[3,2] >= -800
 stack_after_delivery_constr[8,4,0]: - 800 x[8,9] + 400 y[4,0] - s[8,0]
   + s[9,0] >= -800
 stack_after_delivery_constr[8,4,1]: - 800 x[8,9] + 400 y[4,1] - s[8,1]
   + s[9,1] >= -800
 stack_after_delivery_constr[8,4,2]: - 800 x[8,9] + 400 y[4,2] - s[8,2]
   + s[9,2] >= -800
 stack_after_delivery_constr[8,5,0]: - 800 x[8,10] + 400 y[5,0] - s[8,0]
   + s[10,0] >= -800
 stack_after_delivery_constr[8,5,1]: - 800 x[8,10] + 400 y[5,1] - s[8,1]
   + s[10,1] >= -800
 stack_after_delivery_constr[8,5,2]: - 800 x[8,10] + 400 y[5,2] - s[8,2]
   + s[10,2] >= -800
 stack_after_delivery_constr[9,1,0]: - 800 x[9,6] + 400 y[1,0] + s[6,0]
   - s[9,0] >= -800
 stack_after_delivery_constr[9,1,1]: - 800 x[9,6] + 400 y[1,1] + s[6,1]
   - s[9,1] >= -800
 stack_after_delivery_constr[9,1,2]: - 800 x[9,6] + 400 y[1,2] + s[6,2]
   - s[9,2] >= -800
 stack_after_delivery_constr[9,2,0]: - 800 x[9,7] + 400 y[2,0] + s[7,0]
   - s[9,0] >= -800
 stack_after_delivery_constr[9,2,1]: - 800 x[9,7] + 400 y[2,1] + s[7,1]
   - s[9,1] >= -800
 stack_after_delivery_constr[9,2,2]: - 800 x[9,7] + 400 y[2,2] + s[7,2]
   - s[9,2] >= -800
 stack_after_delivery_constr[9,3,0]: - 800 x[9,8] + 400 y[3,0] + s[8,0]
   - s[9,0] >= -800
 stack_after_delivery_constr[9,3,1]: - 800 x[9,8] + 400 y[3,1] + s[8,1]
   - s[9,1] >= -800
 stack_after_delivery_constr[9,3,2]: - 800 x[9,8] + 400 y[3,2] + s[8,2]
   - s[9,2] >= -800
 stack_after_delivery_constr[9,4,0]: - 800 x[9,9] + 400 y[4,0] >= -800
 stack_after_delivery_constr[9,4,1]: - 800 x[9,9] + 400 y[4,1] >= -800
 stack_after_delivery_constr[9,4,2]: - 800 x[9,9] + 400 y[4,2] >= -800
 stack_after_delivery_constr[9,5,0]: - 800 x[9,10] + 400 y[5,0] - s[9,0]
   + s[10,0] >= -800
 stack_after_delivery_constr[9,5,1]: - 800 x[9,10] + 400 y[5,1] - s[9,1]
   + s[10,1] >= -800
 stack_after_delivery_constr[9,5,2]: - 800 x[9,10] + 400 y[5,2] - s[9,2]
   + s[10,2] >= -800
 stack_after_delivery_constr[10,1,0]: - 800 x[10,6] + 400 y[1,0] + s[6,0]
   - s[10,0] >= -800
 stack_after_delivery_constr[10,1,1]: - 800 x[10,6] + 400 y[1,1] + s[6,1]
   - s[10,1] >= -800
 stack_after_delivery_constr[10,1,2]: - 800 x[10,6] + 400 y[1,2] + s[6,2]
   - s[10,2] >= -800
 stack_after_delivery_constr[10,2,0]: - 800 x[10,7] + 400 y[2,0] + s[7,0]
   - s[10,0] >= -800
 stack_after_delivery_constr[10,2,1]: - 800 x[10,7] + 400 y[2,1] + s[7,1]
   - s[10,1] >= -800
 stack_after_delivery_constr[10,2,2]: - 800 x[10,7] + 400 y[2,2] + s[7,2]
   - s[10,2] >= -800
 stack_after_delivery_constr[10,3,0]: - 800 x[10,8] + 400 y[3,0] + s[8,0]
   - s[10,0] >= -800
 stack_after_delivery_constr[10,3,1]: - 800 x[10,8] + 400 y[3,1] + s[8,1]
   - s[10,1] >= -800
 stack_after_delivery_constr[10,3,2]: - 800 x[10,8] + 400 y[3,2] + s[8,2]
   - s[10,2] >= -800
 stack_after_delivery_constr[10,4,0]: - 800 x[10,9] + 400 y[4,0] + s[9,0]
   - s[10,0] >= -800
 stack_after_delivery_constr[10,4,1]: - 800 x[10,9] + 400 y[4,1] + s[9,1]
   - s[10,1] >= -800
 stack_after_delivery_constr[10,4,2]: - 800 x[10,9] + 400 y[4,2] + s[9,2]
   - s[10,2] >= -800
 stack_after_delivery_constr[10,5,0]: - 800 x[10,10] + 400 y[5,0] >= -800
 stack_after_delivery_constr[10,5,1]: - 800 x[10,10] + 400 y[5,1] >= -800
 stack_after_delivery_constr[10,5,2]: - 800 x[10,10] + 400 y[5,2] >= -800
 stack_LIFO_constr[1,0]: - 400 y[1,0] - s[1,0] + s[6,0] >= -800
 stack_LIFO_constr[1,1]: - 400 y[1,1] - s[1,1] + s[6,1] >= -800
 stack_LIFO_constr[1,2]: - 400 y[1,2] - s[1,2] + s[6,2] >= -800
 stack_LIFO_constr[2,0]: - 400 y[2,0] - s[2,0] + s[7,0] >= -800
 stack_LIFO_constr[2,1]: - 400 y[2,1] - s[2,1] + s[7,1] >= -800
 stack_LIFO_constr[2,2]: - 400 y[2,2] - s[2,2] + s[7,2] >= -800
 stack_LIFO_constr[3,0]: - 400 y[3,0] - s[3,0] + s[8,0] >= -800
 stack_LIFO_constr[3,1]: - 400 y[3,1] - s[3,1] + s[8,1] >= -800
 stack_LIFO_constr[3,2]: - 400 y[3,2] - s[3,2] + s[8,2] >= -800
 stack_LIFO_constr[4,0]: - 400 y[4,0] - s[4,0] + s[9,0] >= -800
 stack_LIFO_constr[4,1]: - 400 y[4,1] - s[4,1] + s[9,1] >= -800
 stack_LIFO_constr[4,2]: - 400 y[4,2] - s[4,2] + s[9,2] >= -800
 stack_LIFO_constr[5,0]: - 400 y[5,0] - s[5,0] + s[10,0] >= -800
 stack_LIFO_constr[5,1]: - 400 y[5,1] - s[5,1] + s[10,1] >= -800
 stack_LIFO_constr[5,2]: - 400 y[5,2] - s[5,2] + s[10,2] >= -800
 start_at_depot_constr: u[0] = 0
 empty_stacks_at_depot_constr[0]: s[0,0] = 0
 empty_stacks_at_depot_constr[1]: s[0,1] = 0
 empty_stacks_at_depot_constr[2]: s[0,2] = 0
Bounds
 u[0] <= 10
 u[1] <= 10
 u[2] <= 10
 u[3] <= 10
 u[4] <= 10
 u[5] <= 10
 u[6] <= 10
 u[7] <= 10
 u[8] <= 10
 u[9] <= 10
 u[10] <= 10
 s[0,0] <= 800
 s[0,1] <= 800
 s[0,2] <= 800
 s[1,0] <= 800
 s[1,1] <= 800
 s[1,2] <= 800
 s[2,0] <= 800
 s[2,1] <= 800
 s[2,2] <= 800
 s[3,0] <= 800
 s[3,1] <= 800
 s[3,2] <= 800
 s[4,0] <= 800
 s[4,1] <= 800
 s[4,2] <= 800
 s[5,0] <= 800
 s[5,1] <= 800
 s[5,2] <= 800
 s[6,0] <= 800
 s[6,1] <= 800
 s[6,2] <= 800
 s[7,0] <= 800
 s[7,1] <= 800
 s[7,2] <= 800
 s[8,0] <= 800
 s[8,1] <= 800
 s[8,2] <= 800
 s[9,0] <= 800
 s[9,1] <= 800
 s[9,2] <= 800
 s[10,0] <= 800
 s[10,1] <= 800
 s[10,2] <= 800
Binaries
 x[0,0] x[0,1] x[0,2] x[0,3] x[0,4] x[0,5] x[0,6] x[0,7] x[0,8] x[0,9]
 x[0,10] x[1,0] x[1,1] x[1,2] x[1,3] x[1,4] x[1,5] x[1,6] x[1,7] x[1,8]
 x[1,9] x[1,10] x[2,0] x[2,1] x[2,2] x[2,3] x[2,4] x[2,5] x[2,6] x[2,7]
 x[2,8] x[2,9] x[2,10] x[3,0] x[3,1] x[3,2] x[3,3] x[3,4] x[3,5] x[3,6]
 x[3,7] x[3,8] x[3,9] x[3,10] x[4,0] x[4,1] x[4,2] x[4,3] x[4,4] x[4,5]
 x[4,6] x[4,7] x[4,8] x[4,9] x[4,10] x[5,0] x[5,1] x[5,2] x[5,3] x[5,4]
 x[5,5] x[5,6] x[5,7] x[5,8] x[5,9] x[5,10] x[6,0] x[6,1] x[6,2] x[6,3]
 x[6,4] x[6,5] x[6,6] x[6,7] x[6,8] x[6,9] x[6,10] x[7,0] x[7,1] x[7,2]
 x[7,3] x[7,4] x[7,5] x[7,6] x[7,7] x[7,8] x[7,9] x[7,10] x[8,0] x[8,1]
 x[8,2] x[8,3] x[8,4] x[8,5] x[8,6] x[8,7] x[8,8] x[8,9] x[8,10] x[9,0]
 x[9,1] x[9,2] x[9,3] x[9,4] x[9,5] x[9,6] x[9,7] x[9,8] x[9,9] x[9,10]
 x[10,0] x[10,1] x[10,2] x[10,3] x[10,4] x[10,5] x[10,6] x[10,7] x[10,8]
 x[10,9] x[10,10] y[0,0] y[0,1] y[0,2] y[1,0] y[1,1] y[1,2] y[2,0] y[2,1]
 y[2,2] y[3,0] y[3,1] y[3,2] y[4,0] y[4,1] y[4,2] y[5,0] y[5,1] y[5,2]
 y[6,0] y[6,1] y[6,2] y[7,0] y[7,1] y[7,2] y[8,0] y[8,1] y[8,2] y[9,0]
 y[9,1] y[9,2] y[10,0] y[10,1] y[10,2]
Generals
 u[0] u[1] u[2] u[3] u[4] u[5] u[6] u[7] u[8] u[9] u[10] s[0,0] s[0,1]
 s[0,2] s[1,0] s[1,1] s[1,2] s[2,0] s[2,1] s[2,2] s[3,0] s[3,1] s[3,2]
 s[4,0] s[4,1] s[4,2] s[5,0] s[5,1] s[5,2] s[6,0] s[6,1] s[6,2] s[7,0]
 s[7,1] s[7,2] s[8,0] s[8,1] s[8,2] s[9,0] s[9,1] s[9,2] s[10,0] s[10,1]
 s[10,2]
End
