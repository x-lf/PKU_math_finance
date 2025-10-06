(1). 

为方便推导，这里做一些方便的记法，

在无摩擦的情况下任何债券组合都可以看做面值为1的零息债券的线性组合，计到期时间为$T_i$的这样的债券在t时刻的价格为$p(t,T_i)$,那么债券组合在s时刻的价值为


$$
V_t = \sum_{i=1}^d\lambda_ip(t\Delta,T_i)
$$

另外我们记$y(s,T_i)$为$T_i$到期的债券的收益率并作为风险因子， 则

$$
p(s,T_i) = e^{-(T_i-s)(y(s,T_i))}\\
X_{t+1,i} = \Delta y(s,T_i)
$$

因此有

$$
V_t = \sum_{i=1}^d\lambda_ip(t\Delta,T_i)=\sum_{i=1}^d\lambda_ie^{-(T_i-t\Delta)(y(t\Delta,T_i))}
$$

求导做一阶近似得

$$
L^\Delta_{t+1} = \sum_{i=1}^d\lambda_ie^{-(T_i-t\Delta)(y(t\Delta,T_i))}(y(t\Delta,T_i)\Delta-(T_i-t\Delta)X_{t+1,i})
$$
