from matplotlib import pyplot as plt
import numpy as np
import random


def noeud_terminal(history):
    #on check si le noeud est terminal
    b = history in ['bp', 'bb', 'pp', 'pbb', 'pbp']
    return b

def get_gains(history, cards):
    if history in ['bp', 'pbp']:
        #si un joueur s'est couché
        gains = 1
        return gains
    else:  
    # pp ou bb ou pbb (deux passes ou deux mises successives):
        if 'b' in history :
            #deux mises successives
            gains = 2
        else :
            #deux passes successives
            gains = 1
        current_player = len(history) % 2
        my_card = cards[current_player]
        opponent_card = cards[(current_player + 1) % 2]
        if my_card == 'K' or opponent_card == 'J':
            return gains
        else:
            return -gains


# parier et call(bet) ,  check et se coucher(pass):
possible_actions = ['b', 'p']  

class InfoSet():
    def __init__(self):
        self.strategie_sum = np.zeros(len(possible_actions))
        self.regret_sum = np.zeros(len(possible_actions))
        self.n_actions = len(possible_actions)

    def get_strategie(self, r_p):
        strategie = np.maximum(0, self.regret_sum)
        # on normalise
        if sum(strategie) > 0 :
            strategie /= sum(strategie)
        else :
            strategie =np.repeat(1/self.n_actions, self.n_actions)
        self.strategie_sum += r_p * strategie
        return strategie

    def get_average_strategie(self):
        #on normalise
        if sum(self.strategie_sum) > 0 :
            return self.strategie_sum/sum(self.strategie_sum)
        else :
            return np.repeat(1/self.n_actions, self.n_actions)
        

class CFR():
    def __init__(self):
        # on initialise le dictionnaire des ensembles d'informations
        self.InfoSetMap = {} 

    def get_infoset(self, card_history):
        #avoir l'ensemble d'information
        if card_history not in self.InfoSetMap:
            self.InfoSetMap[card_history] = InfoSet()
        return self.InfoSetMap[card_history]

    def cfr(self, cards, history, r_proba, current_player):
    ## fonction récursive pour parcourir tous les noeuds possibles
        #si on est dans un noeud terminal on return les gains et on arrete la recursivité du programme:
        if noeud_terminal(history):
            return get_gains(history, cards)

        my_card = cards[current_player]
        info_set = self.get_infoset(my_card + history)

        strategie = info_set.get_strategie(r_proba[current_player])
        opponent = (current_player + 1) % 2
        counterfact_act = np.zeros(len(possible_actions))

        for i, action in enumerate(possible_actions):
            action_proba = strategie[i]

            new_r_proba = r_proba.copy()
            new_r_proba[current_player] *= action_proba

            #on appelle la fonction cfr récursivement
            counterfact_act[i] = -1 * self.cfr(cards, history + action, new_r_proba, opponent)

        valeur_noeud = counterfact_act.dot(strategie)
        for i, action in enumerate(possible_actions):
            info_set.regret_sum[i] += r_proba[opponent] * (counterfact_act[i] - valeur_noeud)

        return valeur_noeud

    def entrainement(self, nombre_iterations):
        #entraienement du modele
        val = 0
        P1 = []
        P2 = []
        P1K = []
        P1Q = []
        P1J = []
        iter = []
        deck = ['K', 'Q', 'J']
        for i in range(nombre_iterations):
            #shuffle deck:
            cards = random.sample(deck, 2)
            history = ''
            r_proba = np.ones(2)
            val += self.cfr(cards, history, r_proba, 0)
            # pour tracer la courbe des probabilités de parier : 
            """if i>10 :
                iter.append(i)
                for hist, I in cfr.InfoSetMap.items():
                    if hist == 'K':
                        P1K.append(I.get_average_strategie()[0])
                    elif hist == 'Q':
                        P1Q.append(I.get_average_strategie()[0])
                    elif hist == 'J':
                        P1J.append(I.get_average_strategie()[0])"""
            #pour tracer la courbe valeurs de jeu :
            """if i % 10 == 0:
                iter.append(i)
                P1.append(val/nombre_iterations)
                P2.append(-(val/nombre_iterations))
                #print(val/nombre_iterations)"""
        return val,iter,P1K,P1Q,P1J,P1,P2

if __name__ == "__main__":

    nombre_iterations = 100000

    #arrondir : 
    np.set_printoptions(precision=2, floatmode='fixed', suppress=True)

    cfr = CFR()

    val,iter,P1K,P1Q,P1J,P1,P2 = cfr.entrainement(nombre_iterations)

    #affichage des listes de proba et des iter:
    """print ('iter---------------')
    print(iter)
    print(P1K)
    print(P1Q)"""

    #affichage des valeurs de jeu
    print(f"Valeur de jeu du Player 1    :  {round(val / nombre_iterations, 3)}\n")
    print(f"Valeur de jeu du Player 2    :  {round(-(val / nombre_iterations), 3)}\n")

    # pour filtrer les strategies de chaque joueur:
    P1strat = []
    P2strat = []
    for hist, I in sorted(cfr.InfoSetMap.items(), key=lambda x: len(x[0]) ) : 
        if len(hist) % 2 == 0 : 
            P2strat.append((hist,I.get_average_strategie()))
        else :
            P1strat.append((hist,I.get_average_strategie()))


    #print les strat du premier joueur
    print("\nPlayer 1 strategies [Bet , pass]:\n")
    for (hist, I) in P1strat : 
        print(f"{hist:5}:    {I}")
    
    #print les strat du deuxième joueur
    print("\nPlayer 2 strategies [Bet , pass]:\n")
    for (hist, I) in P2strat : 
        print(f"{hist:5}:    {I}")
    
    
    # tracer la courbe des probabilités de parier :
    """plt.plot(iter,P1K,'g', label = 'King')
    plt.plot(iter,P1Q,'r', label = 'Queen')
    plt.plot(iter,P1J,'b', label = 'Jack')
    plt.ylabel('Fréquence de parier')
    plt.xlabel('Nb Iterations')
    plt.title('Variation de la fréquence de parier en fonction du nombre d iterations ')
    plt.legend()
    plt.show()"""



    #tracer la courbe des valeur de jeu
    """plt.plot(iter,P1,'b',label = 'Joueur 1')
    plt.plot(iter,P2,'g', label = 'Joueur 2')
    plt.ylabel('Valeur de jeu attendue')
    plt.xlabel('Nb Iterations')
    plt.title('Variation de la valeur du jeu en fonction du nombre  d iterations ')
    plt.legend()
    plt.show()"""