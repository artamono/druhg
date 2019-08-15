WIP:
this is a draft of main ideas and properties

Let x and y be some objects.  
Let x<sub>n</sub> is the nth nearest neighbor of x, and y<sub>m</sub> is the mth nearest neighbor of y.  
Then, the ranks(x,y) are n and m if x<sub>n</sub> = y and y<sub>m</sub> = x.  
Then, relatives is |Set of n first nearest neighbors of x ∩ Set of m first nearest neighbors of y|  
Then, ranked distance rd(x,y) is max[dis(x,x<sub>**m**</sub>), dis(y,y<sub>**n**</sub>)]  

Similarity measure between x and y can be defined as min(n,m)/(max(n,m)+relatives) * dis(x,y)/rd(x,y) * 1/rd(x,y)
It can work with ordinal data too by skipping distances.

(1) min(n,m)/(max(n,m)+relatives) ∊ [0, 1]  
(2) dis(x,y)/rd(x,y) ∊ [1/2, 1]  

We can build MST tree with Similarity measure.
