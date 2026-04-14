import numpy as np
from scipy.optimize import fsolve

"def de la fonction sigmoid"
def sigmoid(k, a, r0, L):                                       
    return L / (1 + np.exp(a*(k-r0)))

"fonction pour résoudre (7) cf latex et trouver a, gestion de la non convergence"
def resoudre_a(p_sampled, n, r0, L):
    target = n * p_sampled
    ranks = np.arange(1, n + 1)
    def objective(a):
        return np.sum(sigmoid(ranks, a, r0, L)) - target
    try:
        sol = fsolve(objective, x0=0.5, maxfev=40)                    #fsolve utilise une variante de la méthode hybride de Powell                                                                   
        return float(sol[0])                                            # il peut y avoir des pb de convergence
    except:
        return 0.5

"""
permet de générer la proba d'un élève d'avoir X ou ENS, pour cela on réalise n_sim estimations permettant de calculer
un intervalle de confiance
"""
def simulation_admission_proba(rank_k, p_1an, p_5ans, n, score_daur, max_daur, min_daur, n_sim=10):
    ratio = (score_daur - min_daur)/(max_daur - min_daur)
    L = np.clip(ratio, 0.5, 1.0)
    mu = p_5ans
    sigma = abs(p_5ans - p_1an)               
    probas_at_k = []
    for _ in range(n_sim):
        p_sim = np.random.normal(mu, sigma)
        p_sim = np.clip(p_sim, 0.001, 0.999)   # on clip pour éviter les abérrations
        r0 = n * p_sim 
        a_sim = resoudre_a(p_sim, n, r0, L)    # appel du solver pour a 
        p_k = sigmoid(rank_k, a_sim, r0, L)
        probas_at_k.append(p_k)
    return np.array(probas_at_k)    
