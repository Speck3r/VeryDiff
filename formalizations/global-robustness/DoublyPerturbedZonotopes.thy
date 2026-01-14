theory DoublyPerturbedZonotopes
  imports Complex_Main
begin

text\<open>
We consider 1-dimensional Zonotopes (a.k.a. affine forms).
\<close>

definition zonotope1d :: "real \<Rightarrow> real list \<Rightarrow> real set"
  where "zonotope1d c gs = {
          c + sum_list (map2 (*) s gs) |
          s. (length s = length gs) \<and>
          (\<forall> x \<in> set s . -1 \<le> x \<and> x \<le> 1 )}"


text\<open>
We also consider 1-dimensional Differential Zonotopes.
We model the difference explicitly in this case,
i.e. (x,y) is in the differential zonotope concretization
if (x,y,x-y) is in the construct below.
\<close>

definition diffzono :: "real \<Rightarrow> real \<Rightarrow> real \<Rightarrow> real list \<Rightarrow> real list \<Rightarrow> real list \<Rightarrow> (real\<times>real\<times>real) set"
  where "diffzono c1 c2 cd g1 g2 gd = {
    ((c1 + sum_list (map2 (*) s g1)), (c2 + sum_list (map2 (*) s g2)), (cd + sum_list (map2 (*) s gd))) |
    s . (length s = length g1) \<and>
        (\<forall> x \<in> set s . -1 \<le> x \<and> x \<le> 1 )
  }"

text\<open>
Simple sanity check
\<close>

lemma lets_get_started:
  assumes "l \<le> x \<and> x \<le> u"
      and "l < u"
  shows "x \<in> (zonotope1d ((u+l)/2) [((u-l)/2)])"
  unfolding zonotope1d_def
  apply simp
  apply (intro exI[where x="[(x-(u+l)/2)/((u-l)/2)]"])
  using assms
  apply auto
    apply argo
   apply (simp add: field_simps)
  by (simp add: field_simps)

text\<open>
Soundness of input representation (part 1):
All x1,x2 inside the bounds with difference at most epsilon are within
the doubly-perturbed zonotope that we construct.
\<close>

lemma diff_zono_containment:
  assumes "l \<le> x1 \<and> x1 \<le> u"
      and "l \<le> x2 \<and> x2 \<le> u"
      and "-eps \<le> x1 - x2 \<and> x1 - x2 \<le> eps"
      and "0 < eps \<and> eps < (u-l)"
    defines "g1 \<equiv> ((u-l)/2) - eps/2"
        and "c \<equiv> ((u+l)/2)"
        and "g2 \<equiv> eps/2"
        and "g3 \<equiv> eps/2"
      shows "(x1, x2, (x1-x2)) \<in> (diffzono c c 0.0 [g1, g2, 0.0] [g1, 0.0, g3] [0.0, g2, (-g3)])"
proof -
  (* abbreviations *)
  let ?d = "x1 - x2"
  have d_bounds: "-eps \<le> ?d" " ?d \<le> eps" using assms by simp_all

  (* we will parametrize s2 = t, solve for s1 and s3 *)
  let ?t0 = "((x1 - c) - g1) / g2"   (* lower bound for t from s1 \<ge> -1 *)
  let ?u0 = "(g1 + x1 - c) / g2"     (* upper bound for t from s1 \<le> 1 *)

  (* interval for t ensuring s1 \<in> [-1,1] is [?t0, ?u0] *)
  have s1_interval:
    "\<forall> t. ?t0 \<le> t \<longrightarrow> t \<le> ?u0 \<longrightarrow> -1 \<le> (x1 - c - t*g2)/g1 \<and> (x1 - c - t*g2)/g1 \<le> 1"
    using assms
    by (auto simp: field_simps)

  (* interval for t ensuring s3 \<in> [-1,1]; because g2 = g3 this simplifies *)
  have s3_interval:
    "\<forall> t. (-1 + ?d/g2) \<le> t \<longrightarrow> t \<le> (1 + ?d/g2) \<longrightarrow> -1 \<le> (t*g2 - ?d)/g3 \<and> (t*g2 - ?d)/g3 \<le> 1"
    using assms
    by (auto simp: field_simps)

  (* show the two t-intervals intersect *)
  have "?u0 \<ge> -1 + ?d/g2"
  proof -
    have "?u0 - (-1 + ?d/g2) = (g1 + x1 - c + g2 - ?d) / g2"
      using assms
      by (auto simp: field_simps)
    also have "x1 - c \<ge> - (u - l) / 2"
      using assms
      by (auto simp: field_simps)
    hence "x1 - c + g1 + g2 - ?d \<ge> - (u-l)/2 + (u-l)/2 - ?d"
      using assms
      by (auto simp: field_simps)
    hence "x1 - c + g1 + g2 - ?d \<ge> - ?d"
      using assms
      by (auto simp: field_simps)
    thus ?thesis
      using assms
      by (auto simp: field_simps)
  qed

  have "1 + ?d/g2 \<ge> ?t0"
  proof -
    have "1 + ?d/g2 - ?t0 = (g2 + ?d - (x1 - c) + g1) / g2"
      using assms
      by (auto simp: field_simps)
    also have "- (x1 - c) \<ge> - (u - l) / 2"
      using assms
      by (auto simp: field_simps)
    hence "(g2 + ?d - (x1 - c) + g1) \<ge> g2 + ?d - (u-l)/2 + g1"
      using assms
      by (auto simp: field_simps)
    then have "(g2 + ?d - (x1 - c) + g1) \<ge> 0"
      using assms
      by (auto simp: field_simps)
    thus ?thesis
      using assms
      by (auto simp: field_simps)
  qed

  (* define lb and ub as the intersection endpoints, then pick t *)
  define lb where "lb = max (?t0) (-1 + ?d/g2)"
  define ub where "ub = min (?u0) (1 + ?d/g2)"
  define t  where "t  = max lb (-1)"

  (* prove lb \<le> ub from the two inequalities above *)
  have lb_le_ub: "lb \<le> ub"
  proof -
    have "?t0 \<le> ?u0" using assms g1_def g2_def c_def
      apply (auto simp: field_simps)
      by (metis comm_semiring_class.distrib mult_le_cancel_right_pos order_less_le ordered_comm_semiring_class.comm_mult_left_mono ring_class.ring_distribs(1) zero_less_numeral)
    moreover have "-1 + ?d/g2 \<le> 1 + ?d/g2" by simp
    moreover from `?u0 \<ge> -1 + ?d/g2` have "-1 + ?d/g2 \<le> ?u0" .
    moreover from `1 + ?d/g2 \<ge> ?t0` have "?t0 \<le> 1 + ?d/g2" .
    ultimately show ?thesis
      unfolding lb_def ub_def by (auto simp: max_def min_def)
  qed

  (* show ub \<ge> -1 and lb \<le> 1 so that [lb,ub] intersects [-1,1] *)
  have "ub \<ge> -1"
    unfolding ub_def min_def ub_def g1_def g2_def c_def
    using assms
    by (auto simp: field_simps; linarith)

  have "lb \<le> 1"
    unfolding lb_def max_def lb_def g1_def g2_def c_def
    using assms by (auto simp: field_simps; linarith)

  (* now t = max lb (-1) satisfies -1 \<le> t \<le> 1 and t \<le> ub *)
  have "-1 \<le> t" and "t \<le> 1" and "t \<le> ub"
    unfolding t_def lb_def ub_def
    using lb_le_ub `ub \<ge> -1` `lb \<le> 1`
      apply argo
    using \<open>lb \<le> 1\<close> lb_def apply argo
  using \<open>- 1 \<le> ub\<close> lb_def lb_le_ub ub_def by argo

  (* Now t is in the intersection, so s1 and s3 computed from t are in [-1,1]. *)
  define s2 where "s2 = t"
  define s1 where "s1 = (x1 - c - s2*g2)/g1"
  define s3 where "s3 = (s2*g2 - ?d)/g3"

  have len: "length [s1, s2, s3] = length [g1, g2, 0]" by simp

  have all_bounds: "\<forall>y\<in>set [s1,s2,s3]. -1 \<le> y \<and> y \<le> 1"
  proof (auto)
    show "-1 \<le> s1" and "s1 \<le> 1"
      using s1_interval \<open>t \<le> ub\<close> s2_def s1_def ub_def
      unfolding t_def lb_def ub_def
       apply simp
      using s1_interval \<open>t \<le> ub\<close> s2_def s1_def ub_def
      unfolding t_def lb_def ub_def
      by simp


    show "-1 \<le> s2" and "s2 \<le> 1"
      unfolding s2_def t_def lb_def
      using `-1 \<le> t` `t \<le> 1`
      apply simp
      unfolding s2_def t_def lb_def
      using `-1 \<le> t` `t \<le> 1`
      apply simp
      unfolding s2_def t_def lb_def
      using `-1 \<le> t` `t \<le> 1`
      by simp

    show "-1 \<le> s3" and "s3 \<le> 1"
      using s3_interval s2_def s3_def
      unfolding t_def lb_def ub_def
      using `t \<le> ub`
      apply simp
      using s3_interval s2_def s3_def
      unfolding t_def lb_def ub_def
      using `t \<le> ub`
      apply simp
      using s3_interval s2_def s3_def
      unfolding t_def lb_def ub_def
      using `t \<le> ub`
      apply simp
      using s3_interval s2_def s3_def
      unfolding t_def lb_def ub_def
      using `t \<le> ub`
      by simp
  qed

  (* check the three component equalities *)
  have comp1: "c + sum_list (map2 (*) [s1,s2,s3] [g1,g2,0]) = x1"
    using assms
    by (auto simp: s1_def s2_def s3_def t_def lb_def field_simps)
    

  have comp2: "c + sum_list (map2 (*) [s1,s2,s3] [g1,0,g3]) = x2"
    apply (simp add: s1_def s2_def s3_def t_def)
    apply auto
    using assms
      apply (simp add: field_simps)
    using assms(4) g1_def apply argo
    using assms(4) g3_def by argo    

  have comp3: "0 + sum_list (map2 (*) [s1,s2,s3] [0,g2,-g3]) = x1 - x2"
    apply (simp add: s2_def s3_def)
    apply auto using assms by (simp add: field_simps)

  (* assemble witness and finish *)
  have "\<exists> s. length s = length [g1,g2,0.0] \<and> (\<forall> y\<in>set s. -1 \<le> y \<and> y \<le> 1)
             \<and> c + sum_list (map2 (*) s [g1,g2,0]) = x1
             \<and> c + sum_list (map2 (*) s [g1,0,g3]) = x2
             \<and> 0 + sum_list (map2 (*) s [0,g2,-g3]) = x1 - x2"
    apply (intro exI[where x= "[s1,s2,s3]"])
    using all_bounds comp1 comp2 comp3
    by simp

  then show ?thesis unfolding diffzono_def by auto
qed


text\<open>
Soundness of input representation (part 2):
All (x,y,x-y) inside the doubly perturbed zonotope constructed by us
are inside the bounds [l,u] and are perturbed by at most epsilon:
\<close>

lemma diff_zono_containment_inv:
  fixes u :: "real"
    and l :: "real"
    and eps :: "real"
  defines "g1 \<equiv> ((u-l)/2) - eps/2"
      and "g2 \<equiv> eps/2"
      and "g3 \<equiv> eps/2"
      and "c \<equiv> ((u+l)/2)"
  assumes eps_wf: "0 < eps \<and> eps < (u-l)"
      and in_zono: "(x1, x2, (x1-x2)) \<in> (diffzono c c 0.0 [g1, g2, 0.0] [g1, 0.0, g3] [0.0, g2, (-g3)])"
    shows "l \<le> x1 \<and> x1 \<le> u"
      and "l \<le> x2 \<and> x2 \<le> u"
      and "-eps \<le> x1 - x2 \<and> x1 - x2 \<le> eps"
proof -
  have cur_list_len: "length [(u - l) / 2 - eps / 2, eps / 2, 0 / 10] = 3"
    unfolding List.length_code
    by (simp add: gen_length_code(1) gen_length_code(2))

  have diffzono_rewrite: "(diffzono c c 0.0 [g1, g2, 0.0] [g1, 0.0, g3] [0.0, g2, (-g3)]) = {
       ((c + s1 * ((u - l) / 2 - eps / 2) + s2*(eps / 2)),
       (c + s1 * ((u - l) / 2 - eps / 2) + s3* (eps / 2)),
       (s2 * (eps / 2) + s3 * (- (eps / 2)))) |
       s1 s2 s3 . (
        - 1 \<le> s1 \<and> s1 \<le> 1 \<and>
        - 1 \<le> s2 \<and> s2 \<le> 1 \<and>
        - 1 \<le> s3 \<and> s3 \<le> 1)}"
    using in_zono
    unfolding diffzono_def
    unfolding g1_def g2_def g3_def
    unfolding cur_list_len
    apply simp
  proof (standard)

    show "{(c + sum_list (map2 (*) s [(u - l) / 2 - eps / 2, eps / 2, 0]),
      c + sum_list (map2 (*) s [(u - l) / 2 - eps / 2, 0, eps / 2]),
      sum_list (map2 (*) s [0, eps / 2, - (eps / 2)])) |
     s. length s = 3 \<and> (\<forall>x\<in>set s. - 1 \<le> x \<and> x \<le> 1)}
    \<subseteq> {(c + s1 * ((u - l) / 2 - eps / 2) + s2 * eps / 2,
         c + s1 * ((u - l) / 2 - eps / 2) + s3 * eps / 2,
         s2 * eps / 2 - s3 * eps / 2) |
        s1 s2 s3.
        - 1 \<le> s1 \<and> s1 \<le> 1 \<and> - 1 \<le> s2 \<and> s2 \<le> 1 \<and> - 1 \<le> s3 \<and> s3 \<le> 1}"
    proof auto
      fix s :: "real list"
      assume "length s = 3"
      then obtain s1 s2 s3 where
      s_def: "s = [s1, s2, s3]"
        apply (cases s)
         apply simp
        subgoal for a nexts
          apply (cases nexts)
           apply simp
          subgoal for b nexts
            apply (cases nexts)
             apply simp
            subgoal for c nexts
              apply (cases nexts)
               apply simp
              by simp.
          .
        .
      assume "\<forall>x\<in>set s. - 1 \<le> x \<and> x \<le> 1"
      then have s_range: "- 1 \<le> s1 \<and> s1 \<le> 1 \<and> - 1 \<le> s2 \<and> s2 \<le> 1 \<and> - 1 \<le> s3 \<and> s3 \<le> 1"
        unfolding s_def by simp

      show "\<exists>s1 s2.
            sum_list (map2 (*) s [(u - l) / 2 - eps / 2, eps / 2, 0]) =
            s1 * ((u - l) / 2 - eps / 2) + s2 * eps / 2 \<and>
            (\<exists>s3. sum_list (map2 (*) s [(u - l) / 2 - eps / 2, 0, eps / 2]) =
                  s1 * ((u - l) / 2 - eps / 2) + s3 * eps / 2 \<and>
                  sum_list (map2 (*) s [0, eps / 2, - (eps / 2)]) =
                  s2 * eps / 2 - s3 * eps / 2 \<and>
                  - 1 \<le> s1 \<and>
                  s1 \<le> 1 \<and> - 1 \<le> s2 \<and> s2 \<le> 1 \<and> - 1 \<le> s3 \<and> s3 \<le> 1)"
        unfolding s_def
        apply (rule exI[where x=s1])
        apply (rule exI[where x=s2])
        apply simp
        apply (rule exI[where x=s3])
        by (simp add: s_range)
    qed

    show "{(c + s1 * ((u - l) / 2 - eps / 2) + s2 * eps / 2,
      c + s1 * ((u - l) / 2 - eps / 2) + s3 * eps / 2,
      s2 * eps / 2 - s3 * eps / 2) |
     s1 s2 s3. - 1 \<le> s1 \<and> s1 \<le> 1 \<and> - 1 \<le> s2 \<and> s2 \<le> 1 \<and> - 1 \<le> s3 \<and> s3 \<le> 1}
    \<subseteq> {(c + sum_list (map2 (*) s [(u - l) / 2 - eps / 2, eps / 2, 0]),
         c + sum_list (map2 (*) s [(u - l) / 2 - eps / 2, 0, eps / 2]),
         sum_list (map2 (*) s [0, eps / 2, - (eps / 2)])) |
        s. length s = 3 \<and> (\<forall>x\<in>set s. - 1 \<le> x \<and> x \<le> 1)}"
    proof auto
      fix s1 :: real
      fix s2 :: real
      fix s3 :: real
      assume
        s_ranges:
        "- 1 \<le> s1"
        "s1 \<le> 1"
        " - 1 \<le> s2"
        "s2 \<le> 1"
        "- 1 \<le> s3"
        "s3 \<le> 1"
      show "\<exists>s. s1 * ((u - l) / 2 - eps / 2) + s2 * eps / 2 =
           sum_list (map2 (*) s [(u - l) / 2 - eps / 2, eps / 2, 0]) \<and>
           s1 * ((u - l) / 2 - eps / 2) + s3 * eps / 2 =
           sum_list (map2 (*) s [(u - l) / 2 - eps / 2, 0, eps / 2]) \<and>
           s2 * eps / 2 - s3 * eps / 2 =
           sum_list (map2 (*) s [0, eps / 2, - (eps / 2)]) \<and>
           length s = 3 \<and> (\<forall>x\<in>set s. - 1 \<le> x \<and> x \<le> 1)"
        apply (rule exI[where x="[s1,s2,s3]"])
        by (simp add: s_ranges)
    qed
  qed

  show "l \<le> x1 \<and> x1 \<le> u"
    using in_zono
    unfolding diffzono_rewrite
  proof simp
    assume H:
      "\<exists>s1 s2.
         x1 = c + s1 * ((u - l) / 2 - eps / 2) + s2 * eps / 2 \<and>
         (\<exists>s3. x2 = c + s1 * ((u - l) / 2 - eps / 2) + s3 * eps / 2 \<and>
               x1 - x2 = s2 * eps / 2 - s3 * eps / 2 \<and>
               -1 \<le> s1 \<and> s1 \<le> 1 \<and> -1 \<le> s2 \<and> s2 \<le> 1 \<and> -1 \<le> s3 \<and> s3 \<le> 1)"
    then obtain s1 s2 s3 :: real where
      x1_def:
        "x1 = c + s1 * ((u - l) / 2 - eps / 2) + s2 * eps / 2" and
      bounds:
        "-1 \<le> s1" "s1 \<le> 1" "-1 \<le> s2" "s2 \<le> 1" "-1 \<le> s3" "s3 \<le> 1"
      by blast

    have x1_rewrite:
      "x1 = c + s1 * ((u - l - eps) / 2) + s2 * (eps / 2)"
      using x1_def by algebra

    show ?thesis
      unfolding x1_rewrite
      using eps_wf
      using bounds
      apply (simp)
      apply (intro conjI)
       apply (smt (z3) c_def divide_le_eq_1 field_sum_of_halves mult_minus_left nonzero_mult_div_cancel_right)
      by (smt (z3) c_def divide_le_eq_1 field_sum_of_halves nonzero_mult_div_cancel_right) (* 1.3 seconds *)
  qed

  show "l \<le> x2 \<and> x2 \<le> u"
    using in_zono
    unfolding diffzono_rewrite
  proof simp
    assume H:
      "\<exists>s1 s2.
         x1 = c + s1 * ((u - l) / 2 - eps / 2) + s2 * eps / 2 \<and>
         (\<exists>s3. x2 = c + s1 * ((u - l) / 2 - eps / 2) + s3 * eps / 2 \<and>
               x1 - x2 = s2 * eps / 2 - s3 * eps / 2 \<and>
               -1 \<le> s1 \<and> s1 \<le> 1 \<and> -1 \<le> s2 \<and> s2 \<le> 1 \<and> -1 \<le> s3 \<and> s3 \<le> 1)"
    then obtain s1 s2 s3 :: real where
      x1_def:
        "x2 = c + s1 * ((u - l) / 2 - eps / 2) + s3 * eps / 2" and
      bounds:
        "-1 \<le> s1" "s1 \<le> 1" "-1 \<le> s2" "s2 \<le> 1" "-1 \<le> s3" "s3 \<le> 1"
      by blast

    have x2_rewrite:
      "x2 = c + s1 * ((u - l - eps) / 2) + s3 * (eps / 2)"
      using x1_def by algebra

    show ?thesis
      unfolding x2_rewrite
      using eps_wf
      using bounds
      apply (simp)
      apply (intro conjI)
       apply (smt (z3) c_def divide_le_eq_1 field_sum_of_halves mult_minus_left nonzero_mult_div_cancel_right)
      by (smt (z3) c_def divide_le_eq_1 field_sum_of_halves nonzero_mult_div_cancel_right) (* 1.3 seconds *)
  qed

  show "- eps \<le> x1 - x2 \<and> x1 - x2 \<le> eps"
    using in_zono
    unfolding diffzono_rewrite
  proof simp
    assume H:
      "\<exists>s1 s2.
         x1 = c + s1 * ((u - l) / 2 - eps / 2) + s2 * eps / 2 \<and>
         (\<exists>s3. x2 = c + s1 * ((u - l) / 2 - eps / 2) + s3 * eps / 2 \<and>
               x1 - x2 = s2 * eps / 2 - s3 * eps / 2 \<and>
               -1 \<le> s1 \<and> s1 \<le> 1 \<and> -1 \<le> s2 \<and> s2 \<le> 1 \<and> -1 \<le> s3 \<and> s3 \<le> 1)"
    then obtain s1 s2 s3 :: real where
      x1_def:
        "x1 - x2 = s2 * eps / 2 - s3 * eps / 2" and
      bounds:
        "-1 \<le> s1" "s1 \<le> 1" "-1 \<le> s2" "s2 \<le> 1" "-1 \<le> s3" "s3 \<le> 1"
      by blast

    show ?thesis
      unfolding x1_def
      using eps_wf
      using bounds
      apply (intro conjI)
       apply (smt (z3) c_def divide_le_eq_1 field_sum_of_halves mult_minus_left nonzero_mult_div_cancel_right)
      by (smt (verit, del_insts) divide_le_eq_1_pos field_sum_of_halves mult_minus_left nonzero_mult_div_cancel_right)
  qed
qed

text\<open>
Overall this provides us with the following soundness argument:
\<close>

lemma diff_zono_containment_overall:
  fixes u :: "real"
    and l :: "real"
    and eps :: "real"
  defines "g1 \<equiv> ((u-l)/2) - eps/2"
      and "g2 \<equiv> eps/2"
      and "g3 \<equiv> eps/2"
      and "c \<equiv> ((u+l)/2)"
  assumes eps_wf: "0 < eps \<and> eps < (u-l)"
  shows "
        (x1, x2, (x1-x2)) \<in> (diffzono c c 0.0 [g1, g2, 0.0] [g1, 0.0, g3] [0.0, g2, (-g3)])
        \<longleftrightarrow>
        (l \<le> x1 \<and> x1 \<le> u \<and> l \<le> x2 \<and> x2 \<le> u \<and> -eps \<le> x1 - x2 \<and> x1 - x2 \<le> eps)"
proof
  show "(x1, x2, x1 - x2)
    \<in> diffzono c c (0 / 10) [g1, g2, 0 / 10] [g1, 0 / 10, g3]
        [0 / 10, g2, - g3] \<Longrightarrow>
    l \<le> x1 \<and> x1 \<le> u \<and> l \<le> x2 \<and> x2 \<le> u \<and> - eps \<le> x1 - x2 \<and> x1 - x2 \<le> eps"
    using assms
    using diff_zono_containment_inv
    by simp
  show "l \<le> x1 \<and> x1 \<le> u \<and> l \<le> x2 \<and> x2 \<le> u \<and> - eps \<le> x1 - x2 \<and> x1 - x2 \<le> eps \<Longrightarrow>
    (x1, x2, x1 - x2)
    \<in> diffzono c c (0 / 10) [g1, g2, 0 / 10] [g1, 0 / 10, g3]
        [0 / 10, g2, - g3]"
    using assms
    using diff_zono_containment
    by simp
qed

text\<open>
Some util functions
\<close>

lemma length_3_conv:
"length xs = 3 \<longleftrightarrow> (\<exists>x y z. xs = [x,y,z])"
  apply (cases xs)
   apply simp
  subgoal for a list
    apply (cases list)
     apply simp
    subgoal for aa lista
      apply (cases lista)
       apply simp
      by auto.
  .

definition diffzono_eval :: "real \<Rightarrow> real \<Rightarrow> real \<Rightarrow> real list \<Rightarrow> real list \<Rightarrow> real list \<Rightarrow> real list \<Rightarrow> (real \<times> real \<times> real)" where
"diffzono_eval c1 c2 cd g1 g2 gd s = (
(c1 + sum_list (map2 (*) s g1)),
(c2 + sum_list (map2 (*) s g2)),
(cd + sum_list (map2 (*) s gd)))"

text\<open>
Input splits along input generator as well as along perturbation generators
preserve soundness, i.e. all points are in one of the two resulting zonotopes
\<close>

lemma input_split:
  fixes c g1 g2 g3
  defines "g1_new \<equiv> (g1/2)"
  defines "c_new1 \<equiv> (c - g1_new)"
  defines "c_new2 \<equiv> (c + g1_new)"
  assumes "(x1, x2, (x1-x2)) \<in> (diffzono c c 0.0 [g1, g2, 0.0] [g1, 0.0, g3] [0.0, g2, (-g3)])"
  shows "
    (x1, x2, (x1-x2)) \<in> (diffzono c_new1 c_new1 0.0 [g1_new, g2, 0.0] [g1_new, 0.0, g3] [0.0, g2, (-g3)])
    \<or>
    (x1, x2, (x1-x2)) \<in> (diffzono c_new2 c_new2 0.0 [g1_new, g2, 0.0] [g1_new, 0.0, g3] [0.0, g2, (-g3)])"
proof -
  (* extract witness s from diffzono *)
  from assms
  obtain s where Len: "length s = 3"
    and Sbounds: "(\<forall> y \<in> set s. -1 \<le> y \<and> y \<le> 1)"
    and Eqs: "(c + sum_list (map2 (*) s [g1,g2,0])) = x1
             \<and> (c + sum_list (map2 (*) s [g1,0,g3])) = x2
             \<and> (0 + sum_list (map2 (*) s [0,g2,-g3])) = x1 - x2"
    unfolding diffzono_def
    by force
  (* turn s (length 3) into explicit components *)
  from Len obtain s1 s2 s3 where S_def: "s = [s1,s2,s3]"
  proof -
    have "\<exists>s1 s2 s3. s = [s1,s2,s3]"
      using Len
      apply (cases s)
       apply auto
      subgoal for a list
        apply (cases list)
         apply auto
        subgoal for aa lista
          apply (cases lista)
          by auto.
      .

    then show "(\<And>s1 s2 s3. s = [s1, s2, s3] \<Longrightarrow> thesis) \<Longrightarrow> length s = 3 \<Longrightarrow> thesis"
      by blast
  qed
  hence Sbounds' : " -1 \<le> s1 \<and> s1 \<le> 1 \<and> -1 \<le> s2 \<and> s2 \<le> 1 \<and> -1 \<le> s3 \<and> s3 \<le> 1"
    using Sbounds by auto

  have eq1: "c + s1*g1 + s2*g2 = x1"
    and eq2: "c + s1*g1 + s3*g3 = x2"
    and eq3: "s2*g2 - s3*g3 = x1 - x2"
    using Eqs S_def
    by auto
  (* split on sign of s1 *)
  show ?thesis
  proof (cases "s1 \<le> (0::real)")
    case True
    (* define transformed s1' = 1 + 2*s1 \<in> [-1,1] when s1 \<le> 0 *)
    define s1' where "s1' = 1 + 2 * s1"
    have s1'_bounds: "-1 \<le> s1' \<and> s1' \<le> 1"
      using True Sbounds' unfolding s1'_def by auto
    (* check new equalities with center c_new1 = c - g1/2 and g1_new = g1/2 *)
    have "c_new1 + s1'*g1_new + s2*g2 = x1"
      unfolding c_new1_def g1_new_def s1'_def
      using eq1 by (simp add: algebra_simps)
    moreover have "c_new1 + s1'*g1_new + s3*g3 = x2"
      unfolding c_new1_def g1_new_def s1'_def
      using eq2 by (simp add: algebra_simps)
    moreover have "0 + s2*g2 - s3*g3 = x1 - x2"
      using eq3 by simp
    ultimately have "(x1, x2, (x1-x2)) \<in>
           diffzono c_new1 c_new1 0.0 [g1_new, g2, 0.0] [g1_new, 0.0, g3] [0.0, g2, - g3]"
      unfolding diffzono_def
      apply simp
      apply (intro exI[where x="[s1', s2, s3]"])
      using s1'_bounds
      apply simp
      using Sbounds' by force
    then show ?thesis by auto
  next
    case False
    (* s1 > 0 branch; define s1' = 2*s1 - 1 *)
    define s1' where "s1' = 2 * s1 - 1"
    have s1'_bounds: "-1 \<le> s1' \<and> s1' \<le> 1"
      using False Sbounds' unfolding s1'_def by auto
    (* check equalities with center c_new2 = c + g1/2 *)
    have "c_new2 + s1'*g1_new + s2*g2 = x1"
      unfolding c_new2_def g1_new_def s1'_def
      using eq1 by (simp add: algebra_simps)
    moreover have "c_new2 + s1'*g1_new + s3*g3 = x2"
      unfolding c_new2_def g1_new_def s1'_def
      using eq2 by (simp add: algebra_simps)
    moreover have "0 + s2*g2 - s3*g3 = x1 - x2"
      using eq3 by simp
    ultimately have "(x1, x2, (x1-x2)) \<in>
           diffzono c_new2 c_new2 0.0 [g1_new, g2, 0.0] [g1_new, 0.0, g3] [0.0, g2, - g3]"
      unfolding diffzono_def
      apply simp
      apply (intro exI[where x="[s1', s2, s3]"])
      using s1'_bounds
      apply simp
      using Sbounds' by force
    then show ?thesis by auto
  qed
qed

lemma input_split2:
  fixes c g1 g2 g3
  defines "g2_new \<equiv> (g2/2)"
  defines "cd1 \<equiv> (- g2/2)"
  defines "cd2 \<equiv> ( (g2/2))"
  assumes "(x1, x2, (x1-x2)) \<in> (diffzono c c 0.0 [g1, g2, 0.0] [g1, 0.0, g3] [0.0, g2, (-g3)])"
  shows "
    (x1, x2, (x1-x2)) \<in> (diffzono (c+cd1) c cd1 [g1, g2_new, 0.0] [g1, 0.0, g3] [0.0, g2_new, (-g3)])
    \<or>
    (x1, x2, (x1-x2)) \<in> (diffzono (c+cd2) c cd2 [g1, g2_new, 0.0] [g1, 0.0, g3] [0.0, g2_new, (-g3)])"
proof -
  from assms
  obtain s where Len: "length s = 3"
  and Sbounds: "(\<forall> y \<in> set s. -1 \<le> y \<and> y \<le> 1)"
  and Eqs: "diffzono_eval c c 0.0 [g1,g2,0] [g1,0,g3] [0,g2,-g3] s = (x1,x2,x1-x2)"
    unfolding diffzono_def
    apply simp
    using diffzono_eval_def numeral_3_eq_3 by force
  from Len obtain s1 s2 s3 where S_def: "s = [s1,s2,s3]"
    by (auto simp: length_3_conv)
  have eqs1: "c + s1*g1 + s2*g2 = x1" 
      and eqs2: "c + s1*g1 + s3*g3 = x2"
      and eqs3: "s2*g2 - s3*g3 = x1 - x2"
    using Eqs S_def
    unfolding diffzono_eval_def
    by auto
  
  
  show ?thesis
  proof (cases "s2 \<le> (0::real)")
    case True
    have Sbounds' : " -1 \<le> s1 \<and> s1 \<le> 1 \<and> -1 \<le> (s2+0.5)*2 \<and> (s2+0.5)*2 \<le> 1 \<and> -1 \<le> s3 \<and> s3 \<le> 1"
      using Sbounds S_def True
      by auto
    have "(x1,x2,x1-x2) = diffzono_eval (c - g2/2) c (- g2/2) [g1,g2_new,0] [g1,0,g3] [0,g2_new,-g3] [s1,(s2+0.5)*2,s3]"
      unfolding diffzono_eval_def g2_new_def
      apply (auto simp: field_simps)
      using eqs1 eqs2 eqs3
      apply auto
      using eqs1 apply force
      by argo
      
    then show ?thesis
      unfolding diffzono_def diffzono_eval_def cd1_def
      apply simp
      apply (intro disjI1)
      using Sbounds'
      apply (intro exI[where x="[s1,(s2+0.5)*2,s3]"])
      by auto
    next
    case False
    have Sbounds' : " -1 \<le> s1 \<and> s1 \<le> 1 \<and> -1 \<le> (s2-0.5)*2 \<and> (s2-0.5)*2 \<le> 1 \<and> -1 \<le> s3 \<and> s3 \<le> 1"
      using Sbounds S_def False
      by auto
    have "(x1,x2,x1-x2) = diffzono_eval (c + g2/2) c (g2/2) [g1,g2_new,0] [g1,0,g3] [0,g2_new,-g3] [s1,(s2-0.5)*2,s3]"
      unfolding diffzono_eval_def g2_new_def
      apply (auto simp: field_simps)
      using eqs1 eqs2 eqs3
      apply auto
      using eqs1 apply force
      by argo
      
    then show ?thesis
      unfolding diffzono_def diffzono_eval_def cd2_def
      apply simp
      apply (intro disjI2)
      using Sbounds'
      apply (intro exI[where x="[s1,(s2-0.5)*2,s3]"])
      by auto
  qed
qed

lemma input_split3:
  fixes c g1 g2 g3
  defines "g3_new \<equiv> (g3/2)"
  defines "cd1 \<equiv> (- (g3/2))"
  defines "cd2 \<equiv> ( (g3/2))"
  assumes "(x1, x2, (x1-x2)) \<in> (diffzono c c 0.0 [g1, g2, 0.0] [g1, 0.0, g3] [0.0, g2, (-g3)])"
shows "
    (x1, x2, (x1-x2)) \<in> (diffzono c (c+cd1) (-cd1) [g1, g2, 0.0] [g1, 0.0, g3_new] [0.0, g2, (-g3_new)])
    \<or>
    (x1, x2, (x1-x2)) \<in> (diffzono c (c+cd2) (-cd2) [g1, g2, 0.0] [g1, 0.0, g3_new] [0.0, g2, (-g3_new)])"
proof -
  from assms
  obtain s where Len: "length s = 3"
    and Sbounds: "(\<forall> y \<in> set s. -1 \<le> y \<and> y \<le> 1)"
    and Eqs: "diffzono_eval c c 0.0 [g1,g2,0] [g1,0,g3] [0,g2,-g3] s = (x1,x2,x1-x2)"
    unfolding diffzono_def
    apply simp
    using diffzono_eval_def numeral_3_eq_3 by force

  from Len obtain s1 s2 s3 where S_def: "s = [s1,s2,s3]"
    by (auto simp: length_3_conv)

  have eqs1: "c + s1*g1 + s2*g2 = x1"
    and  eqs2: "c + s1*g1 + s3*g3 = x2"
    and  eqs3: "s2*g2 - s3*g3 = x1 - x2"
    using Eqs S_def
    unfolding diffzono_eval_def
    by auto

  show ?thesis
  proof (cases "s3 \<le> (0::real)")
    case True
    have Sbounds' :
      "-1 \<le> s1 \<and> s1 \<le> 1 \<and> -1 \<le> s2 \<and> s2 \<le> 1 \<and> -1 \<le> (s3+0.5)*2 \<and> (s3+0.5)*2 \<le> 1"
      using Sbounds S_def True
      by auto

    have "(x1,x2,x1-x2) =
      diffzono_eval c (c - g3/2) ( g3/2)
        [g1,g2,0] [g1,0,g3_new] [0,g2,-g3_new]
        [s1,s2,(s3+0.5)*2]"
      unfolding diffzono_eval_def g3_new_def
      apply simp
      apply (intro conjI)
      using eqs1 apply simp
      using eqs2 apply argo
      using eqs3 by argo
      

    then show ?thesis
      unfolding diffzono_def diffzono_eval_def cd1_def
      apply simp
      apply (intro disjI1)
      using Sbounds'
      apply (intro exI[where x="[s1,s2,(s3+0.5)*2]"])
      unfolding cd2_def
      by simp
  next
    case False
    have Sbounds' :
      "-1 \<le> s1 \<and> s1 \<le> 1 \<and> -1 \<le> s2 \<and> s2 \<le> 1 \<and> -1 \<le> (s3-0.5)*2 \<and> (s3-0.5)*2 \<le> 1"
      using Sbounds S_def False
      by auto

    have "(x1,x2,x1-x2) =
      diffzono_eval c (c + g3/2) (-g3/2)
        [g1,g2,0] [g1,0,g3_new] [0,g2,-g3_new]
        [s1,s2,(s3-0.5)*2]"
      unfolding diffzono_eval_def g3_new_def
      apply simp
      apply (intro conjI)
      using eqs1 apply simp
      using eqs2 apply argo
      using eqs3 by argo

    then show ?thesis
      unfolding diffzono_def diffzono_eval_def cd2_def
      apply simp
      apply (intro disjI2)
      using Sbounds'
      apply (intro exI[where x="[s1,s2,(s3-0.5)*2]"])
      by auto
  qed
qed

end