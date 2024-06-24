import pandas as pd
from bs4 import BeautifulSoup
import requests
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 1: Fetching data from Wikipedia
path = 'https://en.m.wikipedia.org/wiki/Trillion_dollar_club_(macroeconomics)'
pathdata = requests.get(path)
soup = BeautifulSoup(pathdata.text, 'html.parser')

# Step 2: Parsing tables from the HTML data
economy = pd.DataFrame(columns=['Trillion Dollar', 'Year', 'Country'])
tables = soup.find_all('table')

# Step 3: Extracting and processing data from each table
for t in tables:
    temp_economy = pd.read_html(StringIO(str(t)))[0]
    caption = t.find_previous('div').find('h3') 
    if caption:
        trillion_dollar_index = caption.get_text(strip=True)
    else:
        continue
    temp_economy.columns = ['Year', 'Country', 'Source']
    if 'Source' in temp_economy.columns:
        temp_economy.drop(columns='Source', inplace=True)
    temp_economy['Trillion Dollar'] = trillion_dollar_index
    economy = pd.concat([economy, temp_economy], ignore_index=True)

# Step 4: Cleaning data and saving to CSV
economy = economy.drop_duplicates()
economy.to_csv("WorldEconomy.csv", index=False)

# Step 5: Setting plot font and style
font_path = fm.findfont(fm.FontProperties(family="Ericsson Hilda"))
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Ericsson Hilda'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelcolor'] = 'navy'
plt.rcParams['xtick.color'] = 'navy'    
plt.rcParams['ytick.color'] = 'navy'

# Step 6: Plotting Trillion Dollar Economies growth by country over time
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")
for country in economy['Country'].unique():
    country_data = economy[economy['Country'] == country]
    plt.plot(country_data['Year'], country_data['Trillion Dollar'], marker='o', label=country)

plt.xlabel('Year', fontsize=9, color='navy')
plt.ylabel('Trillion Dollar Economy', fontsize=9, color='navy')
plt.title('Growth of Trillion Dollar Economies by Country', fontsize=11, fontweight='bold', color='navy')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Country', title_fontsize='13', fontsize='11')
plt.figtext(0.5, 0.01, "* Data webscrapped and Prepared by Samujjal Kalita. Source: Wikipedia link: https://en.m.wikipedia.org/wiki/Trillion_dollar_club_(macroeconomics)", ha="center", fontsize=8, color='navy')
plt.tight_layout()
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
plt.show()

# Step 7: Plotting count of Trillion Dollar milestones reached by each country
plt.figure(figsize=(12, 8))
sns.countplot(y='Country', data=economy, palette='Set2', order=economy['Country'].value_counts().index)
plt.xlabel('Amount of Trillion Dollars', fontsize=9, color='navy')
plt.ylabel('Country', fontsize=9, color='navy')
plt.title('Trillion Dollar Milestones Reached by Country', fontsize=11, fontweight='bold', color='navy')
plt.figtext(0.5, 0.01, "* Data webscrapped and Prepared by Samujjal Kalita. Source: Wikipedia link: https://en.m.wikipedia.org/wiki/Trillion_dollar_club_(macroeconomics)", ha="center", fontsize=8, color='navy')
plt.tight_layout()
ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
plt.show()

# Step 8: Calculating growth rates and plotting them
economy['Trillion Dollar'] = economy['Trillion Dollar'].astype(str).str.replace('US\$| trillion economy', '', regex=True).astype(float)
growth_rates = []
for country in economy['Country'].unique():
    country_data = economy[economy['Country'] == country].sort_values(by='Year')
    initial_value = country_data.iloc[0]['Trillion Dollar']
    final_value = country_data.iloc[-1]['Trillion Dollar']
    num_years = country_data['Year'].max() - country_data['Year'].min()
    if num_years == 0:
        continue
    
    growth_rate = (((final_value / initial_value) ** (1 / num_years)) - 1)*100
    growth_rates.append((country, growth_rate))

growth_rates_df = pd.DataFrame(growth_rates, columns=['Country', 'Growth Rate'])

plt.figure(figsize=(12, 8))
sns.barplot(x='Country', y='Growth Rate', data=growth_rates_df, palette='Set2')
plt.xlabel('Country', fontsize=9, color='navy')
plt.ylabel('Average Annual Growth Rate', fontsize=9, color='navy')
plt.title('Average Annual Growth Rates of Trillion Dollar Economies', fontsize=11, fontweight='bold', color='navy')
plt.xticks(rotation=45)
plt.figtext(0.5, 0.01, "* Data webscrapped and Prepared by Samujjal Kalita. Source: Wikipedia link: https://en.m.wikipedia.org/wiki/Trillion_dollar_club_(macroeconomics)", ha="center", fontsize=8, color='navy')
plt.tight_layout()
plt.show()

# Step 9: Predicting future Trillion Dollar economies for selected countries
countries = ['India', 'China', 'United States']
years = list(range(2030, 2051, 2))

predictions = pd.DataFrame(columns=['Country', 'Year', 'Predicted Trillion Dollar Economy'])

for country in countries:
    country_data = economy[economy['Country'] == country].sort_values(by='Year')
    if len(country_data) < 2:
        continue

    country_data['Year'] = pd.to_numeric(country_data['Year'])
    country_data['Trillion Dollar'] = pd.to_numeric(country_data['Trillion Dollar'])

    X = country_data['Year'].values.reshape(-1, 1)
    y = country_data['Trillion Dollar']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    for year in years:
        predicted_value = model.predict([[year]])
        new_row = pd.DataFrame({'Country': [country], 'Year': [year], 'Predicted Trillion Dollar Economy': [predicted_value[0]]})
        predictions = pd.concat([predictions, new_row], ignore_index=True)

predictions['Year'] = pd.to_numeric(predictions['Year'])
print(predictions)

# Step 10: Visualizing predicted growth of Trillion Dollar economies
sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))
sns.lmplot(x='Year', y='Predicted Trillion Dollar Economy', hue='Country', data=predictions, markers='o', ci=None, scatter_kws={'s':50}, line_kws={'lw':2}, legend=False)

plt.xlabel('Year', fontsize=9, color='navy')
plt.ylabel('Predicted Trillion Dollar Economy', fontsize=9, color='navy')
plt.title('Predicted Growth of Trillion Dollar Economies by Country', fontsize=11, fontweight='bold', color='navy')
plt.figtext(0.5, 0.01, "* Prediction Prepared by Samujjal Kalita. Train and Test data Source: Wikipedia link", ha="center", fontsize=8, color='navy')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Country', title_fontsize='13', fontsize='11')
plt.show()
