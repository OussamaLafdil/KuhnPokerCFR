from matplotlib import pyplot as plt
import random
import numpy as np


def get_card(carte, n_joueurs):
    #get le nom de la carte : nombre -> nom 
    return deck[-n_joueurs - 1:][int(carte[0])] + carte[1:]


def tous_fold(history, n_joueurs):
    return len(history) >= n_joueurs and history.endswith('p' * (n_joueurs - 1))


def get_gains(cards, history, n_joueurs) :
    #return les gains
    player = len(history) % n_joueurs
    player_cards = cards[:n_joueurs]
    num_opponents = n_joueurs - 1
    if history == 'p' * n_joueurs:
        gains = [-1] * n_joueurs
        gains[np.argmax(player_cards)] = num_opponents
        return gains
    elif tous_fold(history, n_joueurs):
        gains = [-1] * n_joueurs
        gains[player] = num_opponents
    else:
        gains = [-1] * n_joueurs
        active_cards = []
        active_indices = []
        for (i, j) in enumerate(player_cards):
            if 'b' in history[i::n_joueurs]:
                gains[i] = -2
                active_cards.append(j)
                active_indices.append(i)
        gains[active_indices[np.argmax(active_cards)]] = len(active_cards) - 1 + num_opponents
    return gains


def noeud_terminal(history, n_joueurs):
    #check si le noeud est terminal
    all_raise = history.endswith('b' * n_joueurs)
    all_acted_after_raise = (history.find('b') > -1) and (len(history) - history.find('b') == n_joueurs)
    all_but_1_player_folds = tous_fold(history, n_joueurs)
    return all_raise or all_acted_after_raise or all_but_1_player_folds


#cartes de jeu
deck = ['1','2','3','4','5','6','7','8','9', 'D', 'J', 'Q', 'K']

# parier et call(bet) ,  check et se coucher(pass):
possible_actions = ['b', 'p']  


class InfoSet():
    def __init__(self):
        self.n_actions = len(possible_actions)
        self.strategie_sum = np.zeros(len(possible_actions))
        self.regret_sum = np.zeros(len(possible_actions))
        
    def get_strategie(self, r_p):
        strategie = np.maximum(0, self.regret_sum)
        #on normalise
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
    def __init__(self, n_joueurs):
        self.InfoSetMap = {}
        self.n_joueurs = n_joueurs
        self.cards = [i for i in range(self.n_joueurs + 1)]

    def get_infoset(self, card_and_history) :
        #get l'ensemble d'information
        if card_and_history not in self.InfoSetMap:
            self.InfoSetMap[card_and_history] = InfoSet()
        return self.InfoSetMap[card_and_history]


    def cfr(self, cards, history, r_proba, current_player):
    #fonction récursive pour parcourir tous les noeuds possibles
        #si noeud terminal:
        if noeud_terminal(history, self.n_joueurs):
            return get_gains(cards, history, self.n_joueurs)

        my_card = cards[current_player]
        info_set = self.get_infoset(str(my_card) + history)

        strategie = info_set.get_strategie(r_proba[current_player])
        opponent = (current_player + 1) % self.n_joueurs

        counterfact_act = [None] * len(possible_actions)

        for i, action in enumerate(possible_actions):
            proba_act = strategie[i]
            new_r_proba = r_proba.copy()
            new_r_proba[current_player] *= proba_act
            #appel recursif de la fct
            counterfact_act[i] = self.cfr(cards, history + action, new_r_proba, opponent)

        val_noeud = strategie.dot(counterfact_act)  

        for i, action in enumerate(possible_actions):
            counterfact_r_proba = np.prod(r_proba[:current_player]) * np.prod(r_proba[current_player + 1:])
            regrets = counterfact_act[i][current_player] - val_noeud[current_player]
            info_set.regret_sum[i] += counterfact_r_proba * regrets
        return val_noeud

    def entrainement(self, nombre_iterations):
        #entraitnement:
        val = np.zeros(self.n_joueurs)
        P1 = []
        P2 = []
        P3 = []
        P4 = []
        P1K = []
        P1Q = []
        P1J = []
        iter = []
        for i in range(nombre_iterations):
            cards = random.sample(self.cards, self.n_joueurs)
            history = ''
            r_proba = np.ones(self.n_joueurs)
            val += self.cfr(cards, history, r_proba, 0)
            #pour tracer la courbe de valeurs de jeu
            """if i % 100 == 0 or i == nombre_iterations-1:
                iter.append(i)
                P1.append(val[0]/nombre_iterations)
                P2.append((val[1]/nombre_iterations))
                P3.append((val[2]/nombre_iterations))
                P4.append((val[3]/nombre_iterations))
                #print(val/nombre_iterations)"""
        return val,iter,P1K,P1Q,P1J,P1,P2,P3,P4




if __name__ == "__main__":
    
    nombre_iterations = 100000


    #on arrondit:
    np.set_printoptions(precision=2, floatmode='fixed', suppress=True)
    

    n_joueurs = 4

    cfr = CFR(n_joueurs)


    val,iter,P1K,P1Q,P1J,P1,P2,P3,P4 = cfr.entrainement(nombre_iterations)
    P1strat = []
    P2strat = []
    P3strat = []
    P4strat = []
 
    #print val de jeu
    for joueur in range(n_joueurs):
        print(f"Valeur de jeu du player {joueur + 1} : {(val[joueur] / nombre_iterations):.3f}")


    #filtre les strat de chaque joueur
    for hist, I in sorted(cfr.InfoSetMap.items(), key=lambda x: len(x[0]) ) : 
        if len(hist) == 1 or len(hist) == 5 : 
            P1strat.append((get_card(hist,n_joueurs),I.get_average_strategie()))
        elif len(hist) == 2 or len(hist) == 6 :
            P2strat.append((get_card(hist,n_joueurs),I.get_average_strategie()))
        elif len(hist) == 3 or len(hist) == 7 :
            P3strat.append((get_card(hist,n_joueurs),I.get_average_strategie()))
        else :
            P4strat.append((get_card(hist,n_joueurs),I.get_average_strategie()))



    #print strat de chaque joueur
    print("\nPlayer 1 strategies [Bet , pass]:\n")
    for (hist, I) in P1strat : 
        print(f"{hist:5}:    {I}")
    
    print("\nPlayer 2 strategies [Bet , pass]:\n")
    for (hist, I) in P2strat : 
        print(f"{hist:5}:    {I}")

    print("\nPlayer 3 strategies [Bet , pass]:\n")
    for (hist, I) in P3strat : 
        print(f"{hist:5}:    {I}")
    
    print("\nPlayer 4 strategies [Bet , pass]:\n")
    for (hist, I) in P4strat : 
        print(f"{hist:5}:    {I}")


    #print les listes de valeur de jeu
    """print(iter)
    print('--------')
    print(P1)
    print('--------')
    print(P2)
    print('--------')
    print(P3)
    print('--------')
    print(P4)
    print('--------')"""


    #plot les valeurs de jeu
    """plt.plot(iter,P1,'b',label = 'Joueur 1')
    plt.plot(iter,P2,'r', label = 'Joueur 2')
    plt.plot(iter,P3,'g', label = 'Joueur 3')
    plt.plot(iter,P4,'k', label = 'Joueur 4')
    plt.ylabel('Valeur de jeu ')
    plt.xlabel('Nb Iterations')
    plt.title('Variation de la valeur du jeu en fonction du nombre d iterations ')
    plt.legend()
    plt.show()"""

