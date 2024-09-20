import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import streamlit.components.v1 as components


df_2024_Table = pd.read_excel("DataForTable2.1.xls")
df_continents = pd.read_csv("continents2.csv")


st.sidebar.title("World Happiness Report : Dataviz' & Prédiction")
pages = ["Contexte", "Jeu de données", "Dataviz'", "Modélisation", "Conclusion"]
page=st.sidebar.radio("Aller vers", pages)

# ///////////////////
# // PAGE CONTEXTE //
# ///////////////////

if page == "Contexte":
    st.title("World Happiness Report - Introduction")
    st.image("images/whr-intro.png", use_column_width=True)
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


# /////////////////////////
# // PAGE JEU DE DONNEES //
# /////////////////////////


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


# ///////////////////////////
# // PAGE DATAVIZ STATIQUE //
# ///////////////////////////


elif page == "Dataviz'":

    df_2024 = pd.read_csv("df_2024_modifie.csv")
    option = st.selectbox(
    'Choisissez un graphique à afficher :',
    ('Nuage de points',
     'Histogramme',
     'Heatmap',
     'Bar Chart Race',
     'Carte Interactive',
     'Diagrammes')
    )

    
    # SCATTERPLOT

    if option == 'Nuage de points':
        
        st.title('Relations entre Life Ladder et les différentes variables')

        variable = st.selectbox("Sélectionnez une variable", [
           'Log GDP per capita', 'Social support', 
           'Healthy life expectancy at birth', 'Freedom to make life choices',
           'Generosity', 'Perceptions of corruption', 'Positive affect', 'Negative affect'])

        fig = px.scatter(df_2024, 
                 x= variable, 
                 y='Life Ladder', 
                 color='Region',
                 title='Relation entre ' + variable + ' et Life Ladder',
                 
                 hover_name='Country name')

        st.plotly_chart(fig)


    # HISTOGRAMME

    elif option == 'Histogramme':

        st.title('Comparaison des valeurs moyennes par région')

        cols_to_convert = ['Life Ladder', 'Log GDP per capita', 'Social support', 
                           'Healthy life expectancy at birth', 'Freedom to make life choices', 
                           'Generosity', 'Perceptions of corruption']

        df_2024[cols_to_convert] = df_2024[cols_to_convert].astype(float)

        # Calculer les moyennes par région, en ignorant les colonnes non numériques
        df_mean = df_2024.groupby('Region')[cols_to_convert].mean().reset_index()

        # Transformation des données pour une visualisation avec plotly.express
        df_long = df_mean.melt(id_vars=['Region'], 
                               value_vars=cols_to_convert,
                               var_name='Variable', 
                               value_name='Value')

        # Créer un graphique en barres empilées pour comparer les valeurs moyennes
        fig = px.bar(df_long, x='Region', y='Value', color='Variable', barmode='group',
                    
                     labels={'Value': 'Valeur moyenne', 'Variable': 'Facteur'})

        # Afficher le graphique dans Streamlit

        st.plotly_chart(fig)


    # HEATMAP

    elif option == 'Heatmap':

        st.title('Heatmap des Corrélations')
    
        # Calcul de la matrice de corrélation
        colonnes_a_convertir = ['Life Ladder', 'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
                            'Freedom to make life choices', 'Generosity','Perceptions of corruption', 'Positive affect', 'Negative affect']
    
        corr = df_2024[colonnes_a_convertir].corr()

        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.xticks(rotation=50, ha='right')
        plt.title('Heatmap des Corrélations')
    
        st.pyplot(plt)           


    # BAR CHART RACE

    elif option == 'Bar Chart Race':

        st.subheader("Bar Chart Race")

        df_2024 = pd.read_csv("df_2024_modifie.csv")

        components.iframe("https://public.flourish.studio/visualisation/19439091/embed", width=700, height=500)

        

    # CARTE INTERACTIVE

    elif option == 'Carte Interactive':

        st.subheader("Carte interactive")

        # Chargement du dataset modifié
        df_2024 = pd.read_csv("df_2024_modifie.csv")

        year = st.slider("Sélectionnez une année", min_value=int(df_2024['Year'].min()), 
                     max_value=int(df_2024['Year'].max()), value=2023)

        variable = st.selectbox("Sélectionnez une variable", [
           'Life Ladder', 'Log GDP per capita', 'Social support', 
           'Healthy life expectancy at birth', 'Freedom to make life choices',
           'Generosity', 'Perceptions of corruption', 'Positive affect', 'Negative affect'])

        df_filtered = df_2024[df_2024['Year'] == year]


        fig = px.choropleth(df_filtered, 
                            locations="Country name", 
                            locationmode="country names", 
                            color=variable, 
                            hover_name="Country name",  
                            hover_data={variable: True},  
                            color_continuous_scale=px.colors.sequential.Cividis, 
                            title=f"Carte du {variable} en {year}")

        fig.update_geos(
           showcoastlines=True, coastlinecolor="Black",  
           showland=True, landcolor="lightgray",  )
        
        st.plotly_chart(fig)


    # DIAGRAMMES EN RADAR

    elif option == 'Diagrammes':
    
        height = st.slider("Ajuster la hauteur pour optimiser l'affichage des filtres sélectionnés (pixels). Puis cliquer sur un filtre pour relancer l'affichage.", min_value=800, max_value=6000, value=6000)
        components.iframe("https://public.flourish.studio/visualisation/19439621/embed", height=height)



# ///////////////////////
# // PAGE MODELISATION //
# ///////////////////////

elif page == "Modélisation":
    st.title("Modélisation")

    # Classification du problème
    st.header("Classification du problème")

    st.write("""
    Notre projet se concentre sur un problème de régression. En effet, nous cherchons à prédire le "Ladder score", 
    une variable continue qui reflète une mesure de bien-être ou de satisfaction de vie. Pour ce faire, nous utilisons des variables explicatives 
    telles que le PIB par habitant, le soutien social, l'espérance de vie en bonne santé, etc.
    """)

    # Métriques de performance
    st.subheader("Métriques de performance")
    st.write("""
    Nous avons utilisé deux métriques principales pour comparer nos modèles : la **Mean Squared Error (MSE)** et la **Root Mean Squared Error (RMSE)**.
    La MSE pénalise plus fortement les grandes erreurs de prédiction, en mettant au carré les écarts entre les valeurs prédites et les valeurs réelles. 
    Cela nous permet d'accorder plus de poids aux erreurs significatives, crucial pour garantir la précision des modèles. 
    La MSE est couramment utilisée dans les algorithmes de régression, facilitant l'optimisation des modèles.

    En parallèle, la **Root Mean Squared Error (RMSE)**, racine carrée de la MSE, est exprimée dans les mêmes unités que la variable cible (le Ladder score), 
    ce qui la rend plus intuitive pour évaluer les erreurs de prédiction.

    En dehors de la MSE et de la RMSE, nous avons également utilisé d'autres métriques pour affiner notre évaluation des performances des modèles :
    - **Mean Absolute Error (MAE)** : calcule l'erreur moyenne en valeur absolue, offrant une mesure plus robuste car moins influencée par les outliers.
    - **Coefficient de détermination (R²)** : mesure la proportion de la variance expliquée par le modèle et nous permet d'évaluer dans quelle mesure nos modèles capturent la variabilité des données.
    
    En combinant ces différentes métriques, nous obtenons une évaluation complète et robuste des performances de nos modèles.
    """)

    # Choix du modèle et optimisation
    st.header("Choix du modèle et optimisation")

    st.write("""
    Nous avons confronté 3 modèles de régression que vous pouvez tester ci-dessous. 
    **XGBoost** s'est révélé être le modèle le plus performant. Nous l'avons hyper-paramétré, notamment pour éviter le sur-aprentissage.         
    """)
    st.subheader("Performances des modèles")
    st.image("images/recap-model-perf.png", width=500)
    


    # Charger les données
    df = pd.read_csv("df_2024_modifie.csv")
    feats = df.drop(['Year', 'Country name', 'Life Ladder'], axis=1)
    target = df['Life Ladder']

# Pre processing des données (train test, OHE, valeurs manquantes...)
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2, random_state = 42)
    OHE = OneHotEncoder(drop='first', sparse_output=False)  
    X_train_encoded = OHE.fit_transform(X_train[['Region', 'Sub region']])
    X_test_encoded = OHE.transform(X_test[['Region', 'Sub region']])
    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=OHE.get_feature_names_out(['Region', 'Sub region']))
    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=OHE.get_feature_names_out(['Region', 'Sub region']))
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True) 
    X_train_full = pd.concat([X_train.drop(['Region', 'Sub region'], axis=1), X_train_encoded_df], axis=1)
    X_test_full = pd.concat([X_test.drop(['Region', 'Sub region'], axis=1), X_test_encoded_df], axis=1)
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    X_train_full_imputed = imputer.fit_transform(X_train_full)
    X_test_full_imputed = imputer.transform(X_test_full)

    all_columns = X_train_full.columns

    X_train = pd.DataFrame(X_train_full_imputed, columns=all_columns)
    X_test = pd.DataFrame(X_test_full_imputed, columns=all_columns)

    model_option = st.selectbox("Sélectionnez un modèle de régression", ["Linear Regression", "Random Forest", "XGBoost [optimisé]"])

# Fonction pour afficher les performances
    def display_performance(model_name, y_test, y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
    
        st.subheader(f"Performances du modèle {model_name}")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        st.write(f"R² Score: {r2:.2f}")

    # Graphique des prédictions
        fig, ax = plt.subplots()
        ax.scatter(y_pred, y_test, color='blue', alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel('Valeurs prédites')
        ax.set_ylabel('Valeurs réelles')
        ax.set_title(f"Graphique des prédictions pour {model_name}")
        st.pyplot(fig)

# Fonction pour afficher les feature importance (top 15) pour Random Forest et Linear Regression
    def display_feature_importance(model, model_name, X_train):
        st.subheader(f"Feature Importance pour {model_name}")
    
        if model_name == "Linear Regression":
           # Pour Linearr Regression
            feature_importance = np.abs(model.coef_)
            feature_importance_df = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)

        elif model_name == "Random Forest":
        # Pour Random Forest
            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)
    
        feature_importance_df = feature_importance_df.head(15)

    # Affichage graphiques Feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Features')
        ax.set_title(f"Top 15 Feature Importance pour {model_name}")
        plt.gca().invert_yaxis()  
        st.pyplot(fig)

# Fonction spécifique pour afficher les feature importance pour XGBoost (top 15)
    def display_feature_importance_xgboost(model):
        st.subheader("Feature Importance pour XGBoost")
        fig, ax = plt.subplots(figsize=(10, 6))
        xgb.plot_importance(model, ax=ax, max_num_features=15)
        plt.title('Top 15 Feature Importance ')
        st.pyplot(fig)


# Modélisation en fonction du choix
    if model_option == "Linear Regression":
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        display_performance("Linear Regression", y_test, y_pred)
        display_feature_importance(lr, "Linear Regression", X_train)

    elif model_option == "Random Forest":
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        display_performance("Random Forest", y_test, y_pred)
        display_feature_importance(rf, "Random Forest", X_train)

    elif model_option == "XGBoost [optimisé]":
        # Utilisation de DMatrix pour XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=list(X_test.columns))

    # Paramètres personnalisés pour XGBoost
        params =  {
            'objective': 'reg:squarederror',
            'max_depth': 7,  
            'learning_rate': 0.1, 
            'subsample': 0.9,
            'colsample_bytree': 0.7,  
            'gamma': 0.2, 
            'reg_alpha': 3.0,  
            'reg_lambda': 7.0,  
            'min_child_weight': 3,  
            'seed': 42
        }
        num_boost_round = 150
        model = xgb.train(params, dtrain, num_boost_round)

        y_pred = model.predict(dtest)
    
    # Affichage des performances et des feature importance
        display_performance("XGBoost", y_test, y_pred)
        display_feature_importance_xgboost(model)

# Hyper paramétrage du modèle XG Boost

    st.subheader("Hyper-Paramétrage du modèle XG Boost")
    st.write("""
    Nous avons effectué une recherche par grille (ou “GridSearch”) pour identifier les valeurs optimales des hyper-paramètres. 
    Nous avons ensuite appliqué ces hyper-paramètres qui ont donné d’excellentes performances sur la prédiction du jeu de test : 

**Mean Absolute Error: 0.25           
Mean Squared Error: 0.11           
R² Score: 0.90            
Mean Absolute Percentage Error (MAPE): 4.93%**
             

Toutefois, **qui dit excellentes performances dit aussi risque de surapprentissage** ou “overfitting”. 

Nous avons donc cherché à mesurer s' il y avait de l’overfitting en comparant le RMSE du jeu d'entraînement et celui du jeu de test grâce à une validation croisée. 

Les résultats ont été sans appel (hélas) avec un **RMSE de 0.03 sur le jeu d’entraînement et de 0.38 sur le jeu de test.**
L’écart est bien trop grand et nous avons cherché à réduire cet écart tout en minimisant la perte de performance sur le jeu de test : **en somme, nous avons recherché le bon équilibre.**

Après cette phase d'ajustement, nous pouvons retenir les points clés suivants :

**Nous avons conservé un MSE très performant (qui passe de 0,11 à 0,15)
L’écart entre le Test RMSE et le Train RMSE est de 25% : il y a certes overfitting mais dans des proportions acceptables.**
             
         
    """)
#Analyse feature importance 

    st.subheader("Analyse de la Feature Importance")
    st.write (""" La feature importance est intéressante puisqu’elle montre que le modèle ne performe pas à partir d’une seule variable qui occuperait une importance démesurée par rapport aux autres. 

Il y a donc moins de risque que les performances du modèle soient biaisées. 

Au contraire, un total de 8 variables ont une importance clé dans la prédiction des performances. 

Certes la variable du PIB (“Log GDP per capita”) est 2 fois plus importante que la variable “Perceptions of corruption” dans les performances du modèle mais cela semble finalement très cohérent.  Un niveau de vie élevé a un impact important sur le bien-être d’une population.

Nous relevons également que le continent d’origine et la région améliorent le modèle mais n’ont qu’une importance très relative dans le score de bonheur. 

Cela constitue en soi une information pertinente et intéressante.



    """)
#INterprétation des résultats
    st.header("Interprétation des résultats")
    st.write (""" Voici pour rappel les excellentes performances finales de notre modèle selon les indicateurs clés : 
                 
**​Mean Absolute Error** : 0.29  *erreur moyenne absolue en unités réelles de la variable cible*
              
**Mean Squared Error** : 0.15  *moyenne des carrés des erreurs entre les valeurs prédites et les valeurs réelles.*

**Root Mean Squared Error** : 0.39 *écart-type des résidus*

**R² Score**: 0.87  *indique à quel point le modèle explique la variance des données* 

**Mean Absolute Percentage Error (MAPE)** : 5.74% *affiche l'erreur en % relatif aux valeurs réelles* 

Les excellentes performances de départ étaient trompeuses puisqu’elles ont révélé un overfitting dangereux pour les prédictions futures du modèle.

Nous avons pu rectifier le tir en ajustant les hyper-paramètres dont certains ont eu un fort impact sur les résultats : 

**max_depth** : définit la profondeur maximale de chaque arbre de décision construit par XGBoost. Une profondeur trop élevée peut conduire à un overfitting car le modèle devient trop complexe. Nous avons pu maintenir une profondeur relativement élevée de 7 qui nous semblait important pour la qualité de l’apprentissage. 

**learning_rate** : contrôle la taille des étapes que le modèle fait à chaque nouvel arbre ajouté. Trop faible, il nécessite plus d’itérations de boosting ; trop élevé, il va converger trop rapidement sur les données d’entraînement. Nous avons utilisé un taux de 0.1 modéré, souvent utilisé en pratique.

**subsample** : le sous-échantillonnage détermine si chaque arbre est formé avec toutes les données disponibles ou une fraction. Nous avons fixé un seuil à 0.9 qui permet au modèle de voir suffisamment de données tout en réduisant l’overfitting. 



    """)


elif page == "Conclusion":
    st.subheader("Conclusion")
    st.write("Co")