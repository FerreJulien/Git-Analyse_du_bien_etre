import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df_2024_Table = pd.read_excel("DataForTable2.1.xls")
df_continents = pd.read_csv("continents2.csv")


st.sidebar.title("World Happiness Report : Dataviz' & Prédiction")
pages = ["Contexte", "Jeu de données", "Dataviz'", "Carte interactive", "Modélisation", "Conclusion"]
page=st.sidebar.radio("Aller vers", pages)

if page == "Contexte":
    st.title("World Happiness Report - Introduction")
    st.image("/Users/Thibault/Desktop/WORKFLOWPYTHON/Git-Analyse_du_bien_etre/images/whr-intro.png", use_column_width=True)
    st.header("Contexte et objectifs")

    # Section 1: Genèse du World Happiness Report
    st.subheader("Génèse du World Happiness Report")
    st.write("""
    En 2011, le Bhoutan propose à l’Assemblée Générale des Nations Unies la résolution 65/309 intitulée 
    "Happiness: Towards a holistic approach to development". Cette résolution invite les gouvernements à 
    “accorder plus d’importance au bonheur et au bien-être dans la mesure du développement social et économique.”

    Suite à l’adoption de cette résolution, le 1er rapport du **World Happiness Report** (WHR) est publié en avril 2012. 
    L’ambition du WHR est d’évaluer chaque année le niveau de bonheur de l'ensemble des pays.

    La majorité des données recueillies proviennent du **Gallup World Poll**, l’enquête la plus complète et la plus vaste au monde. 
    Cette enquête touche plus de 99% de la population adulte mondiale grâce à des sondages annuels, avec des métriques comparables entre les pays.

    Les répondants au sondage sont invités à évaluer leur niveau de bonheur ressenti sur une échelle de 1 à 10, ou « ladder score ». 
    Le rapport intègre **6 facteurs** (PIB, espérance de vie, liberté, générosité, vie sociale, corruption) qui aident à comprendre 
    le ladder score mesuré dans chaque pays.
    """)

    # Section 2: Pourquoi ce sujet est-il d'importance ?
    st.subheader("Pourquoi ce sujet est-il d'importance ?")
    st.write("""
    Si la quête du bonheur existe depuis la nuit des temps (Sénèque écrit dès le 1er siècle après J.C « Tous les hommes recherchent le bonheur »), 
    sa prise en compte par les pouvoirs publics comme un objectif de vie est plus récente.

    Depuis quelques années, on ne compte plus les injonctions au bonheur, les livres de développement personnel, les **happiness managers** 
    et les coachs en bien-être. Derrière cet idéal se cachent des réalités bien différentes selon les régions du monde. Le bonheur est en effet 
    une notion subjective qui revêt des habits différents selon les cultures.

    Expliquer le bonheur à travers seulement 6 facteurs semble à première vue réducteur. Et le bonheur n’est pas non plus ressenti de la même manière 
    selon les âges de la vie.

    Bien que la méthodologie du WHR semble imparfaite, elle n’en reste pas moins dénuée d’intérêt en proposant des données harmonisées à l’échelle mondiale. 
    Il s’agit à ce jour de la seule initiative tentant d’apporter une réponse à cette quête impossible du bonheur.

    L’enquête du WHR donne le pouls global d’un monde qui a de moins en moins confiance en l’avenir. Ce constat est particulièrement vrai chez les jeunes 
    générations qui sont, notamment en France, de plus en plus pessimistes.
    """)

    # Section 3: Objectifs du projet
    st.subheader("Les Objectifs")
    st.write("Voici les 3 principaux challenges que nous allons tenter de relever à travers ce projet :")

    st.markdown("#### 1er challenge : Le bonheur en DataViz")
    st.write("""
    Nous allons représenter visuellement le niveau de bonheur dans le monde et son évolution sur les dernières années. 
    Nous verrons la corrélation des 6 facteurs du WHR sur le niveau de bonheur, avec des distinctions par région.
    
    Nous tenterons d’intégrer d’autres paramètres comme le changement climatique pour mesurer leur impact éventuel sur le niveau de bonheur.
    """)

    st.markdown("#### 2ème challenge : « Prédire » le bonheur")
    st.write("""
    Nous allons entraîner et optimiser un modèle qui soit en mesure de prédire le niveau de bonheur par pays pour l’année suivante 
    avec le plus de justesse possible.
    """)

    st.markdown("#### 3ème challenge : Déterminer la « Recette Magique » du bonheur")
    st.write("""
    Grâce au travail réalisé précédemment, nous allons pouvoir identifier les facteurs les plus influents dans la quête du bonheur, 
    avec des distinctions par région.
    """)
    st.write("""
        Vous trouverez dans les différentes sections :
        - Un aperçu du jeu de données utilisé.
        - Des visualisations pour explorer les tendances du bonheur dans le monde.
        - Une carte interactive pour comparer les pays.
        - Une modélisation pour prédire les scores de bonheur.
    """)


elif page == "Jeu de données":
    st.header("Jeu de données")
# Chargement du dataset modifié
    df_2024 = pd.read_csv("df_2024_modifie.csv")

    st.markdown("""
    **Nous avons utilisé le dataset historique du WHR 2024** nommé _"DataForTable2.1"_. Celui-ci est en libre accès [ici](https://happiness-report.s3.amazonaws.com/2024/DataForTable2.1.xls).

    Ce dataset comporte **2363 entrées** s’étalant de **2005 à 2023**. Nous pouvons ainsi connaître les résultats des **165 pays** ayant participé au moins une fois à l’étude.
    
    **Les données collectées sont les suivantes :**
    - Le ladder score du pays, soit la moyenne des évaluations subjectives du niveau de bonheur par chaque répondant.
    - 6 variables : le PIB, l’espérance de vie en bonne santé, la liberté de choix, la générosité, l’importance des connexions sociales, et la perception du niveau de corruption.
    - 2 évaluations subjectives des émotions ressenties par les répondants le jour précédant l’étude : les **émotions positives** (ou “positive affect”) et les **émotions négatives** (“negative affect”).

    **Points de vigilance relevés sur ce dataset :**
    - La liste des pays est très disparate d’une année sur l’autre, ce qui complique les comparaisons d’une année à l’autre. Elle devient plus régulière à partir du début des années 2010.
    - Les premières années sont plus susceptibles de comporter des **valeurs manquantes**.

    **Nous avons décidé de ne retenir que les 10 dernières années** pour la partie Data Visualisation. Ce choix permet de réduire le nombre de valeurs manquantes et d’améliorer la lisibilité des graphiques avec une liste de pays sensiblement homogène d’une année à l’autre.
    
    **Variable cible** :
    Le modèle de machine learning que nous allons entraîner aura pour objectif de prédire le **ladder score** d'un pays en fonction des différentes variables du dataset.
    """)

    st.markdown("### Pré-processing et Feature Engineering")
    st.markdown("""
    Le prétraitement a d'abord consisté à harmoniser les deux jeux de données avant de les fusionner, en veillant à ce que les **noms des pays** soient uniformes des deux côtés. L'objectif était d'ajouter, dans le dataset _"DataForTable2.1"_, des informations précises sur la **situation géographique** de chaque pays, son continent et sa sous-région. Le Kosovo n'étant pas inclus dans le dataset _"continents2"_, ses informations ont été ajoutées manuellement.

    Ensuite, pour obtenir un dataset avec un minimum de **valeurs manquantes** tout en restant suffisamment complet, nous avons conservé uniquement les données des **dix dernières années**, de 2014 à 2023.
    """)

    st.markdown("### Colonnes du dataset modifié")
    st.write(df_2024.columns)

    st.markdown("""
    Les **valeurs manquantes restantes** seront traitées uniquement dans la phase de Machine Learning. Pour la Data Visualisation, aucune autre transformation ne sera effectuée. Le remplacement des valeurs manquantes sera réalisé via la **médiane**, plus robuste face aux **outliers**, d'abord sur le jeu d'entraînement puis sur le jeu de test, afin d'éviter toute **fuite de données**.

    La transformation des données a été effectuée via un **Robust Scaling** pour assurer la comparabilité des variables et optimiser les performances des algorithmes de machine learning. Cette méthode, elle aussi résistante aux **outliers**, permet de neutraliser l'impact des valeurs extrêmes, garantissant que le modèle capte les tendances réelles et fournit des prédictions fiables.

    **Nous prévoyons également d'appliquer une Analyse en Composantes Principales (ACP)** dans la phase de modélisation. Cette technique de réduction de dimension nous permettra de simplifier le modèle en réduisant le nombre de variables tout en conservant l'essentiel de l'information. Le but étant d’éventuellement améliorer les performances des modèles et limiter le risque de **surapprentissage**.
    """)

# Aperçu du dataset modifié
    st.subheader("Aperçu des données modifiées")
    st.dataframe(df_2024.head())

# Aperçu de l'évolution du Ladder Score par pays
    st.subheader(" Aperçu de l'évolution du Ladder Score par pays")
    # Liste des pays disponibles dans le dataset
    countries = df_2024['Country name'].unique()
    # Filtre interactif pour sélectionner un ou plusieurs pays
    selected_countries = st.multiselect("Sélectionnez un ou plusieurs pays", countries, default=["France", "Japan", "Costa Rica"])
    df_filtered = df_2024[df_2024['Country name'].isin(selected_countries)]
    fig = px.line(df_filtered, x="Year", y="Life Ladder", color="Country name", 
                  title="Évolution du Ladder Score (2014-2023)")
    st.plotly_chart(fig)


# Statistiques descriptives
    st.subheader("Statistiques descriptives")
    st.write(df_2024.describe())


elif page == "Dataviz'":
    st.subheader("Dataviz'")

elif page == "Carte interactive":
    st.subheader("Carte interactive")

elif page == "Modélisation":
    st.subheader("Modélisation")

elif page == "Conclusion":
    st.subheader("Conclusion")
    st.write("Co")