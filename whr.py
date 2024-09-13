import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
    st.subheader("Jeu de données")
    st.write(df_2024_Table.head())

elif page == "Dataviz'":
    st.subheader("Dataviz'")

elif page == "Carte interactive":
    st.subheader("Carte interactive")

elif page == "Modélisation":
    st.subheader("Modélisation")

elif page == "Conclusion":
    st.subheader("Conclusion")
    st.write("Co")