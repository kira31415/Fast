import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import streamlit as st
from scipy.integrate import simpson
st.set_page_config(layout="wide")


# Définir les caractéristiques des modèles
voitures = {
    "Dodge Charger R/T": {
        "masse": 1760.0,        # en kg
        "acceleration": 5.1,  # m/s^2
        "largeur" : 1.95,
        "longueur" : 5.28,
        "hauteur" : 1.35,
        "Cx": 0.38,           # coefficient de traînée
        "Cz": 0.3,            # coefficient de portance
        "mu": 0.1,             # coefficient de frottement
        "acceleration_boost" : 5.1
    },
    "Toyota Supra Mark IV": {
        "masse": 1615,
        "acceleration": 5,
        "largeur" : 1.81,
        "longueur" : 4.51,
        "hauteur" : 1.27,
        "Cx": 0.29,
        "Cz": 0.3,
        "mu": 0.1,
        "acceleration_boost" : 5
    },
    "Chevrolet Yenko Camaro": {
        "masse": 1498,
        "acceleration": 5.3,
        "largeur" : 1.88,
        "longueur" : 4.72,
        "hauteur" : 1.30,
        "Cx": 0.35,
        "Cz": 0.3,
        "mu": 0.1,
        "acceleration_boost" : 5.3
    },
    "Mazda RX-7 FD": {
        "masse": 1385,
        "acceleration": 5.2,
        "largeur" : 1.75,
        "longueur" : 4.3,
        "hauteur" : 1.23,
        "Cx": 0.28,
        "Cz": 0.3,
        "mu": 0.1,
        "acceleration_boost" : 5.2
    },
    "Nissan Skyline GTR-R34": {
        "masse": 1540,
        "acceleration": 5.8,
        "largeur" : 1.79,
        "longueur" : 4.6,
        "hauteur" : 1.36,     
        "Cx": 0.34,
        "Cz": 0.3,
        "mu": 0.1,
        "acceleration_boost" : 5.8
    },
    "Mitsubishi Lancer Evolution VII": {
        "masse": 1600,
        "acceleration": 5,
        "largeur" : 1.81,
        "longueur" : 4.51,
        "hauteur" : 1.48,        
        "Cx": 0.28,
        "Cz": 0.3,
        "mu": 0.1,
        "acceleration_boost" : 5
    }
}

# Définir les accessoires
accessoires = {
    "booster_NOS": {
        "acceleration_gain": 0.3  # gain en m/s^2
    },
    "ailerons_jupe_avant": {
        "Cz_gain": 0.1,            # augmentation du Cz
        "Cx_reduction": 0.05,      # réduction du Cx
        "masse": 45                # en kg
    }
}

# Interface utilisateur Streamlit
st.title('Simulation de Vitesse de Voiture')
st.sidebar.header('Choisissez une voiture')

# Choisir un modèle de voiture
voiture_select = st.sidebar.selectbox("Sélectionner une voiture", list(voitures.keys()))

# Récupérer les paramètres de la voiture sélectionnée
voiture = voitures[voiture_select]

# Affichage des informations de la voiture
st.write(f"Modèle sélectionné: {voiture_select}")
st.write(f"Masse: {voiture['masse']} kg")
st.write(f"Accélération: {voiture['acceleration']} m/s²")
st.write(f"Coefficient de traînée (Cx): {voiture['Cx']}")
st.write(f"Coefficient de portance (Cz): {voiture['Cz']}")
st.write(f"Coefficient de friction (mu): {voiture['mu']}")

# Choisir des accessoires
st.sidebar.header('Sélectionner des accessoires')
accessoires_select = st.sidebar.multiselect("Sélectionner des accessoires", list(accessoires.keys()))



S = voiture["largeur"]*voiture["hauteur"]  # surface frontale de la voiture en m²
A = voiture["largeur"]*voiture["longueur"]
# Appliquer les accessoires
for acces in accessoires_select:
    if acces == "booster_NOS":
        voiture['acceleration_boost'] = voiture['acceleration'] * (1 + accessoires[acces]["acceleration_gain"])
    elif acces == "ailerons_jupe_avant":
        voiture['Cz'] = voiture['Cz'] + (voiture['Cz'] *  (1+accessoires[acces]["Cz_gain"]))
        voiture['Cx'] = voiture['Cx'] - (voiture['Cx'] * accessoires[acces]["Cx_reduction"])
        voiture['masse'] += accessoires[acces]["masse"]
        S += 0.8

# Paramètres du problème
h = 2  # hauteur en mètres
theta = 3.7 * np.pi / 180  # angle en radians
L = 31  # distance parcourue en mètres
m = voiture['masse']  # masse de la voiture en kg
mu = voiture['mu']  # coefficient de friction
Cx = voiture['Cx']  # coefficient de traînée
Cz = voiture["Cz"]  # Coefficient de portance de la voiture
acceleration = voiture['acceleration_boost']
rho = 1.225  # densité de l'air en kg/m³
S = voiture["largeur"] * voiture["hauteur"]  # surface frontale de la voiture en m²
A = voiture["longueur"] * voiture["largeur"]
# Constantes
g = 9.81  # accélération gravitationnelle en m/s²


# Fonction représentant les forces agissant sur la voiture
def equations_of_motion(t, v):
    # Composantes de la force
    N = m * g * np.cos(theta)  # force normale
    F_gravity = m * g * np.sin(theta)  # force gravitationnelle
    F_friction = mu * N  # force de friction
    F_air = 0.5 * Cx * S * rho * v**2  # force de résistance de l'air
    
    # Equation du mouvement : F = ma, donc dv/dt = (F_gravity - F_friction - F_air) / m
    dvdt = (F_gravity - F_friction - F_air) / m + acceleration
    return dvdt

# Conditions initiales : vitesse initiale v0 = 0 m/s
v0 = 0

# Résolution numérique de l'équation différentielle
time_points = np.linspace(0, 4, 1000)  # temps de simulation
solution = integrate.odeint(equations_of_motion, v0, time_points)

# Calcul de la distance parcourue à chaque instant en fonction de la vitesse
distance_travelled = np.cumsum(solution[:, 0]) * (time_points[1] - time_points[0])  # distance = vitesse * temps

# Trouver le temps nécessaire pour parcourir la distance L
time_to_reach_L = time_points[np.argmin(np.abs(distance_travelled - L))]
v_at_L = solution[np.argmin(np.abs(distance_travelled - L)), 0]

# Affichage des résultats
st.write(f"Temps nécessaire pour atteindre le bas de la pente : {time_to_reach_L:.2f} secondes")
st.write(f"Vitesse de la voiture au moment d'atteindre {L} m : {v_at_L:.2f} m/s")

# Graphique de la distance parcourue en fonction du temps
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Premier graphique : Distance parcourue en fonction du temps
ax[0].plot(time_points, distance_travelled, label='Distance parcourue')
ax[0].axhline(y=L, color='r', linestyle='--', label=f"Distance L = {L} m")
ax[0].set_title("Distance parcourue en fonction du temps")
ax[0].set_xlabel("Temps (s)")
ax[0].set_ylabel("Distance parcourue (m)")
ax[0].legend()
ax[0].grid(True)

# Deuxième graphique : Vitesse en fonction du temps
ax[1].plot(time_points, solution[:, 0], label='Vitesse')
ax[1].scatter(time_to_reach_L, v_at_L, color='r', zorder=5)  # Marqueur du point
ax[1].text(time_to_reach_L + 0.2, v_at_L, f'({time_to_reach_L:.2f}s, {v_at_L:.2f} m/s)', color='r')
ax[1].set_title("Vitesse en fonction du temps")
ax[1].set_xlabel("Temps (s)")
ax[1].set_ylabel("Vitesse (m/s)")
ax[1].legend()
ax[1].grid(True)

# Affichage des graphiques
st.pyplot(fig)



# Interface utilisateur Streamlit
st.title('Simulation de Vitesse dans le Looping')
st.write(f"Surface frontale: {A} m²")


R = 6     # Rayon du looping (m)
v0 = v_at_L   # Vitesse initiale (m/s)


# Calcul de la vitesse en fonction de l'angle avec frottements, accélération moyenne et résistance de l'air
def vitesse(theta):
    # Accélération tangentielle due à la gravité, accélération moyenne et frottement de l'air
    v = np.zeros_like(theta)
    v[0] = v0
    for i in range(1, len(theta)):
        dtheta = theta[i] - theta[i - 1]
        
        # Calcul de la force de traînée (résistance de l'air)
        F_air = 0.5 * Cx * rho * A * v[i-1]**2 / m
        
        # Accélération totale (gravité, frottement, accélération moyenne, résistance de l'air)
        a_tang = voiture["acceleration"] - 2.75 - mu * g - g * np.sin(theta[i]) - F_air / (v[i-1] if v[i-1] != 0 else 1)

        # Intégration pour obtenir la vitesse
        v[i] = np.sqrt(v[i - 1]**2 + 2 * a_tang * R * dtheta)
        if v[i] < 0:
            v[i] = 0

    return v




# Angle en radians pour un tour complet
theta = np.linspace(0, 2 * np.pi, 1000)  # De 0 à 2*pi radians (un tour complet)

# Calcul des vitesses
v = vitesse(theta)



# Calcul du temps total
def temps_total(theta, v):
    dt = R / v  # Temps élémentaire pour chaque segment
    return simpson(dt)  # Intégration numérique pour le temps total

time = temps_total(theta, v) / 300


st.write(f"Temps pour faire le looping : {time:.2f} s")
# Limite minimale de vitesse dans un looping sans frottement
v_min = np.sqrt(R * g)  # Vitesse minimale requise (m/s)

# Affichage des résultats
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(theta, v, label="Vitesse avec frottements, accélération moyenne et résistance de l'air", color="blue")
ax.axhline(y=v_min, color="red", linestyle="--", label=f"Limite minimale: {v_min:.2f} m/s")
ax.set_title("Évolution de la vitesse dans le looping (avec frottements, accélération moyenne et résistance de l'air)")
ax.set_xlabel("Angle (radians)")
ax.set_ylabel("Vitesse (m/s)")
ax.scatter(theta[-1], v[-1], color='green', zorder=5)
ax.text(theta[-1] + 0.1, v[-1], f'({theta[-1]:.2f}, {v[-1]:.2f} m/s)', color='green')
ax.grid(True)
ax.legend()

# Affichage dans Streamlit
st.pyplot(fig)





# Interface utilisateur Streamlit
st.title('Simulation de la Trajectoire lors du Ravin ')


# Paramètres du problème

v_x0 = v[-1]          # Vitesse initiale en x (m/s)
v_y0 = 0           # Vitesse initiale en y (m/s)
x0, y0 = 0, 1      # Positions initiales en x et y


# Fonction pour les équations différentielles avec frottements et portance
def chute_libre_frottements_portance(z, t):
    vx, vy, x, y = z
    v = np.sqrt(vx**2 + vy**2)  # Vitesse totale
    
    # Forces de traînée
    D_x = -0.5 * Cx * rho * S * v * vx / m
    D_y = -0.5 * Cx * rho * A * v * vy / m

    # Forces de portance
    L = 0.5 * Cz * rho * A* v**2 / m  # Portance totale
    theta = np.arctan2(vy, vx)      # Angle de la vitesse
    L_x = -L * np.sin(theta)        # Portance en x (composante horizontale)
    L_y = L * np.cos(theta)         # Portance en y (composante verticale)

    # Accélérations résultantes
    ax = D_x + L_x
    ay = D_y + L_y - g

    return [ax, ay, vx, vy]  # Retourne les dérivées

# Conditions initiales
z0 = [v_x0, v_y0, x0, y0]

# Temps de simulation
t = np.linspace(0, 1, 500)  # Temps de 0 à 5 secondes

# Résolution des équations différentielles
sol = integrate.odeint(chute_libre_frottements_portance, z0, t)

# Extraction des solutions
vx = sol[:, 0]
vy = sol[:, 1]
x = sol[:, 2]
y = sol[:, 3]

# Arrêter proprement à y = 0 en interpolant
for i in range(len(y)):
    if y[i] <= 0:  # Interpolation pour stopper exactement à y = 0
        ratio = -y[i-1] / (y[i] - y[i-1])
        x_exact = x[i-1] + ratio * (x[i] - x[i-1])
        y_exact = 0
        x = np.append(x[:i], x_exact)
        y = np.append(y[:i], y_exact)
        break

# Dernier point de la trajectoire
x_end = x[-1]
y_end = y[-1]

# Tracé des résultats
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(x, y, label="Trajectoire avec portance et frottements", color="blue")  # Courbe principale

# Ajout du point final de la trajectoire
ax.scatter(x_end, y_end, color='red', s=25, label=f'Fin ({x_end:.2f}, {y_end:.2f})')
ax.text(x_end+1, y_end-0.2, f'({x_end:.2f}, {y_end:.2f})', fontsize=10, color='green', ha='right')

# Ajout de la ligne horizontale de y = 0
ax.plot([9, 50], [0, 0], color='black', linestyle='-', linewidth=2, label="Piste d'atterrissage")
ax.plot([-50, 0], [1, 1], color='black', linestyle='-', linewidth=2,)

# Paramètres du graphique
ax.set_xlabel('Position x (m)')
ax.set_ylabel('Position y (m)')
ax.set_title('Trajectoire de la voiture avec portance et frottements')
ax.set_xlim(-5, 16)  # Ajuste les limites de l'axe x
ax.set_ylim(-1, max(y) + 1)  # Ajuste l'axe y pour qu'il soit dynamique

# Ajouter une légende
ax.legend()

# Affichage dans Streamlit
st.pyplot(fig)

# Trouver l'indice où y atteint zéro
t_impact = t[np.where(y <= 0)[0][0]]  # Temps où la voiture atteint le sol

# Résultats finaux
st.write(f"Position finale : ({x_end:.2f}, {y_end:.2f})")
st.write(f"Temps total : {t_impact:.3f} secondes")



# Interface utilisateur Streamlit
st.title('Temps pour finir le circuit')

# Conditions initiales
v0 = v[-1]  # Vitesse initiale en m/s
x0 = 0  # Position initiale en m
t = np.linspace(0, 2, 1000)  # Temps de 0 à 2 secondes

# Fonction pour l'ODE
def model(y, t):
    v, x = y
    dvdt = - (Cx * rho * S * v**2) / (2 * m) - mu * g + voiture['acceleration']  # Accélération
    dxdt = v  # Vitesse est la dérivée de la position
    return [dvdt, dxdt]

# Conditions initiales
initial_conditions = [v0, x0]

# Résolution de l'ODE
sol = integrate.odeint(model, initial_conditions, t)

# Extraction des résultats
x_sol = sol[:, 1]

dist = x_end - 9

# Distance cible
distance_target = 10 - dist  # Distance à parcourir en mètres

# Temps nécessaire pour parcourir la distance de 10m
time_to_reach = np.interp(distance_target, x_sol, t)
st.write(f"Temps pour parcourir {distance_target:.2f} mètres : {time_to_reach:.2f} secondes")


st.sidebar.header("Le temps total pour faire le circuit")
temps = time_to_reach_L + time_to_reach + t_impact + time
st.sidebar.write(f"{temps:.3f} s")