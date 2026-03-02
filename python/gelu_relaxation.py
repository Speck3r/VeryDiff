
import math
import torch 
from auto_LiRPA.operators.gelu import BoundGelu


class BoundGeluTight(BoundGelu):

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr=attr, inputs=inputs, output_index=output_index, options=options)
        # Number of binary search iterations to find slope bounds
        # TODO: set via options
        self.n_iter = 10  

    @torch.no_grad()
    def find_valid_lower_tangent_bounds(self, l, u, n_iter=10):
        shape = l.shape 
        device = l.device
        dtype = l.dtype

        # --- lower limit of tangent point ---
        # range is [-sqrt(2), x_min], where we use -0.74 as a point a bit right of the min
        lx0 = torch.full(shape, -math.sqrt(2), device=device, dtype=dtype)
        ux0 = torch.full(shape, -0.74, device=device, dtype=dtype)

        for i in range(n_iter):
            x0 = 0.5 * (lx0 + ux0)
            # check if the line at x0 is below gelu at l 
            y = self.d_act_func(x0) * (l - x0) + self.act_func(x0)
            mask = y <= self.act_func(l)

            lx0 = torch.where(mask, lx0, x0)
            ux0 = torch.where(mask, x0, ux0)

        lower_tangent_limit = ux0


        # --- upper limit of tangent point ---
        # range is [x_min, sqrt(2)] where we use 0.74 as a point a bit left of the min
        lx0 = torch.full(shape, -0.76, device=device, dtype=dtype)
        ux0 = torch.full(shape, math.sqrt(2), device=device, dtype=dtype)

        for i in range(n_iter):
            x0 = 0.5 * (lx0 + ux0)
            # check if line at x0 is below gelu at u
            y = self.d_act_func(x0) * (u - x0) + self.act_func(x0)
            mask = y <= self.act_func(u)
            lx0 = torch.where(mask, x0, lx0)
            ux0 = torch.where(mask, ux0, x0)

        upper_tangent_limit = lx0 

        return torch.clamp(lower_tangent_limit, l, u), torch.clamp(upper_tangent_limit, l, u)

    @torch.no_grad()
    def find_valid_upper_left_tangent_bounds(self, l, u, n_iter=10):
        shape = l.shape 
        device = l.device
        dtype = l.dtype

        lx0 = l.clone()
        ux0 = torch.full(shape, -math.sqrt(2), device=device, dtype=dtype)
        # ux0 = torch.max(lx0, ux0)

        for i in range(n_iter):
            x0 = 0.5 * (lx0 + ux0)
            y = self.d_act_func(x0)*(u - x0) + self.act_func(x0)
            mask = y >= self.act_func(u)
            lx0 = torch.where(mask, x0, lx0)
            ux0 = torch.where(mask, ux0, x0)

        return lx0

    @torch.no_grad()
    def find_valid_upper_right_tangent_bounds(self, l, u, n_iter=10):
        lx0 = l.clone()
        # TODO: min of u and sqrt(2)???
        ux0 = u.clone()

        for i in range(n_iter):
            x0 = 0.5 * (lx0 + ux0)
            y = self.d_act_func(x0)*(l - x0) + self.act_func(x0)
            mask = y >= self.act_func(l)
            lx0 = torch.where(mask, lx0, x0)
            ux0 = torch.where(mask, x0, ux0)

        return ux0
    
    @torch.no_grad()
    def find_best_lower_relax_tangent(self, l, u, dl, du):
        """
        find_best_lower_relax_tangent(l, u, dl, du)

        Finds initial tangent point s.t. enclosed area between gelu and lower relaxation is minimal.
        
        :param l: concrete lower bound on the approximation interval
        :param u: concrete upper bound on the approximation interval
        :param dl: concrete lower bound on the valid tangent points
        :param du: concrete upper bound on the valid tangent points
        """
        xm = 0.5 * (l + u)
        # if xm is outside [dl, du], we pick the closest of the valid endpoints
        x0 = torch.clamp(xm, min=dl, max=du)
        return x0
    
    @torch.no_grad()
    def find_best_upper_left_relax_tangent(self, l, u, dl, du):
        xm = 0.5 * (l + u)
        x0 = torch.clamp(xm , min=dl, max=du)
        return x0
    
    @torch.no_grad()
    def find_best_upper_right_relax_tangent(self, l, u, dl, du):
        xm = 0.5 * (l + u)
        x0 = torch.clamp(xm , min=dl, max=du)
        return x0
    
    def opt_init(self):
        super().opt_init()
        self.tp_lower = {}
        self.tp_upper_left = {}
        self.tp_upper_right = {}

    def _init_opt_parameters_impl(self, size_spec, name_start):
        l = self.inputs[0].lower
        shape = [size_spec] + list(l.shape)
        
        alpha = torch.empty(3, *shape, device=l.device, dtype=l.dtype)
        alpha.data[0] = self.tp_lower[name_start].expand(1, *shape)
        alpha.data[1] = self.tp_upper_left[name_start].expand(1, *shape)
        alpha.data[2] = self.tp_upper_right[name_start].expand(1, *shape)

        return alpha

    
    def _bound_relax_params(self, lower, upper, func, dfunc, n_iter=10):       
        y_l, y_u = func(lower), func(upper)
        k_direct = (y_u - y_l) / (upper - lower).clamp(min=1-8)

        sqrt2 = math.sqrt(2)
        concave_left    =  upper <= -sqrt2
        convex_middle   = (upper > -sqrt2) & (upper <= sqrt2) & (lower >= -sqrt2)
        mid_increasing  = (upper > -sqrt2) & (upper <= sqrt2) & (lower <  -sqrt2) & (self.d_act_func(lower) <= k_direct)
        mid_decreasing1 = (upper > -sqrt2) & (upper <= sqrt2) & (lower <  -sqrt2) & (self.d_act_func(lower) >  k_direct) & (self.d_act_func(upper) >  k_direct)
        mid_decreasing2 = (upper > -sqrt2) & (upper <= sqrt2) & (lower <  -sqrt2) & (self.d_act_func(lower) >  k_direct) & (self.d_act_func(upper) <= k_direct)
        concave_right   = (upper > -sqrt2) & (upper >  sqrt2) & (lower >=  sqrt2)
        outer           = (upper > -sqrt2) & (upper >  sqrt2) & (lower <   sqrt2) & (self.d_act_func(upper) >= k_direct)
        positive1       = (upper > -sqrt2) & (upper >  sqrt2) & (lower <   sqrt2) & (self.d_act_func(upper) <  k_direct) & (self.d_act_func(lower) <  k_direct)
        positive2       = (upper > -sqrt2) & (upper >  sqrt2) & (lower <   sqrt2) & (self.d_act_func(upper) <  k_direct) & (self.d_act_func(lower) >= k_direct)

        lower_dl, lower_du = self.find_valid_lower_tangent_bounds(lower, upper, n_iter=n_iter)
        upper_du = self.find_valid_upper_left_tangent_bounds(lower, upper, n_iter=n_iter)
        upper_dl = self.find_valid_upper_right_tangent_bounds(lower, upper, n_iter=n_iter)

        if self.opt_stage in ['opt', 'reuse']:
            ns = self._start 
            self.alpha[ns].data[0] = torch.where(convex_middle | mid_increasing | mid_decreasing1 | outer | positive1, 
                                                 torch.clamp(self.alpha[ns][0], lower_dl, lower_du),
                                                 self.alpha[ns][0])
            
            self.alpha[ns].data[1] = torch.where(concave_left,
                                                 torch.clamp(self.alpha[ns][1], lower, upper),
                                                 torch.where(mid_decreasing1 | mid_decreasing2,
                                                             torch.clamp(self.alpha[ns][1], lower, upper_du),
                                                             self.alpha[ns][1]))
            
            self.alpha[ns].data[2] = torch.where(concave_right,
                                                 torch.clamp(self.alpha[ns][2], lower, upper),
                                                 torch.where(positive1 | positive2,
                                                             torch.clamp(self.alpha[ns][2], upper_dl, upper),
                                                             self.alpha[ns][2]))

            lower_x0 = self.alpha[ns][0]
            upper_left_x0 = self.alpha[ns][1]
            upper_right_x0 = self.alpha[ns][2]

        else:
            xm = 0.5 * (lower + upper)
            lower_best = self.find_best_lower_relax_tangent(lower, upper, lower_dl, lower_du)
            upper_left_best = self.find_best_upper_left_relax_tangent(lower, upper, lower, upper_du)
            upper_right_best = self.find_best_upper_right_relax_tangent(lower, upper, upper_dl, upper)

            # TODO: maybe we can get rid of one torch.where?
            lower_x0 = lower_best.clone()
            lower_x0 = torch.where(concave_left | mid_decreasing2, lower_dl, lower_x0)
            lower_x0 = torch.where(mid_increasing | outer | positive1, lower_best, lower_x0)
            lower_x0 = torch.where(concave_right | positive2, lower_du, lower_x0)

            upper_left_x0 = xm.clone()
            upper_left_x0 = torch.where(convex_middle | mid_increasing | concave_right | outer | positive1 | positive2, upper_du, upper_left_x0)
            upper_left_x0 = torch.where(mid_decreasing1 | mid_decreasing2, upper_left_best, upper_left_x0)

            upper_right_x0 = xm.clone()
            upper_right_x0 = torch.where(concave_left | convex_middle | mid_increasing | mid_decreasing1 | mid_decreasing2 | outer, upper_dl, upper_right_x0)
            upper_right_x0 = torch.where(positive1 | positive2, upper_right_best, upper_right_x0)

            if self.opt_stage == 'init':
                ns = self._start 
                self.tp_lower[ns] = lower_x0 
                self.tp_upper_left[ns] = upper_left_x0
                self.tp_upper_right[ns] = upper_right_x0


        # better to return al, bl, au, bu so we can actually check and debug the relaxation
        # lower relaxations
        direct_lower = concave_left | mid_decreasing2 | concave_right | positive2
        al = torch.where(direct_lower, k_direct, dfunc(lower_x0))
        bl = torch.where(direct_lower, func(lower) - al*lower, func(lower_x0) - al*lower_x0)

        # upper relaxations
        au = torch.where(concave_left | mid_decreasing1 | mid_decreasing2, dfunc(upper_left_x0), k_direct)
        au = torch.where(concave_right | positive1 | positive2, dfunc(upper_right_x0), au)
        bu = torch.where(concave_left | mid_decreasing1 | mid_decreasing2, func(upper_left_x0) - au*upper_left_x0, func(lower) - au*lower)
        bu = torch.where(concave_right | positive1 | positive2, func(upper_right_x0) - au*upper_right_x0, bu)

        return al, bl, au, bu
    
    def bound_relax_impl(self, x, func, dfunc):
        lower, upper = x.lower, x.upper

        al, bl, au, bu = self._bound_relax_params(lower, upper, func, dfunc, n_iter=self.n_iter)
        mask = torch.full(al.shape, True)
        # they use k*(x - x0) + y0 <= f(x)
        # so we want al*x + bl == al*(x - l) + y0 for some y0
        # <--> y0 == al*x + bl - al*(x - l)
        #         == al*x + bl - al*x + al*l
        #         == bl + al*l
        self.add_linear_relaxation(mask=mask, type='lower', k=al, x0=lower, y0=bl + al*lower)
        # same as for lower relaxation
        self.add_linear_relaxation(mask=mask, type='upper', k=au, x0=lower, y0=bu + au*lower)









