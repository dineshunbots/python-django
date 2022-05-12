from django.shortcuts import render
from django.views.generic.base import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin

from tkinter import CENTER
import streamlit as st
import pandas as pd
import numpy as np
from apyori import apriori
from datetime import datetime
from mlxtend.frequent_patterns import apriori, association_rules
from PIL import Image
from io import BytesIO
import xlsxwriter
import base64
import matplotlib.pyplot as pltnew
import seaborn as sns
#from st_aggrid import AgGrid
#import plotly.express as px


# In[15]:

# image = Image.open('WhatsApp-Image-2021-12-23-at-3.05.01-PM-_1_.jpg')
# image1=Image.open('WhatsApp-Image-2021-12-23-at-3.05.01-PM-_2_.jpg')
# first,center,last=st.columns(3)
# first.image(image)
# center.write("")
# last.image(image1)
#df=pd.read_csv("Market basket analysis data.csv")
df=pd.read_csv("Market basket analysis data.csv",on_bad_lines='skip')
#df.Number
df.tail(30)

df['PRICE_RANGE'] = pd.cut(x=df["NETVALUECC1"], bins=[0,2000,10000,20000,30000,40000,50000,60000,75000,600000],labels=['0-2000', '2000-10000', '10000-20000',"20000-30000","30000-40000","40000-50000","50000-60000","60000-75000","75000 and above"])
    #pd.cut(df['some_col'], bins=[0,20,40,60], labels=['0-20', '20-40', '40-60']) 
df=df.dropna()
# def st_csv_download_button(df):
#     csv = df.to_csv(index=False) #if no filename is given, a string is returned
#     b64 = base64.b64encode(csv.encode()).decode()
#     href = f'<a href="data:file/csv;base64,{b64}">Download Sample Template</a>'
#     st.sidebar.markdown(href, unsafe_allow_html=True)  

#st_csv_download_button(data)    
#check1 = st.sidebar.button("sample template")
#st.set_wide_mode()
# st.markdown(""" <style> .font {
# font-size:50px ; font-family: 'calibri'; color: #0094cb;} 
# </style> """, unsafe_allow_html=True)
# st.markdown('<p class="font">Market Basket Analysis</p>', unsafe_allow_html=True)
#st.title('Market Basket Analysis')
#st.markdown(""" <style> .font {
#font-size:11px ; font-family: 'calibri'; color: #f48220;} 
#</style> """, unsafe_allow_html=True)
#st.markdown('<p class="font">Market basket analysis is a data mining technique used by retailers to increase sales by better understanding customer purchasing patterns. It involves analyzing large data sets, such as purchase history, to reveal product groupings, as well as products that are likely to be purchased together.</p>', unsafe_allow_html=True)
#st.write("Market basket analysis is a data mining technique used by retailers to increase sales by better understanding customer purchasing patterns. It involves analyzing large data sets, such as purchase history, to reveal product groupings, as well as products that are likely to be purchased together.")
#st.write("----------------------------------------------------------------------------------------")
#st.write("An association rule has two parts: an **antecedent** (if) and a **consequent** (then). An antecedent is an item found within the data. A consequent is an item found in combination with the antecedent. ... Association rules are calculated from itemsets, which are made up of two or more items.")
#st.write("**consequent**  : item found in combination with the antecedent")
#st.write("**support**     : Support is an indication of how frequently the items appear in the data. It refers to how often a given rule appears in the database being mined.")
#st.write("**confidence**  : Confidence indicates the number of times the if-then statements are found true.Confidence refers to the amount of times a given rule turns out to be true in practice. A rule may show a strong correlation in a data set because it appears very often but may occur far less when applied. This would be a case of high support, but low confidence.Conversely, a rule might not particularly stand out in a data set, but continued analysis shows that it occurs very frequently. This would be a case of high confidence and low support. Using these measures helps analysts separate causation from correlation, and allows them to properly value a given rule. ")
#st.write("**lift**        : lift can be used to compare confidence with expected confidence, or how many times an if-then statement is expected to be found true. It is the ratio of confidence to support. If the lift value is a negative value, then there is a negative correlation between datapoints. If the value is positive, there is a positive correlation, and if the ratio equals 1, then there is no correlation.")
#if check1:
 #       st.write(data.head(10))
new={'PN':"diamond pendant",
'RN':"diamond ring",
'BT':"diamond bracelet",
'ER':"diamond earring",
'BN':"diamond bangle",
'NK':"diamond necklace",
'BR':"gold bracelet",
'GB':"gold bracelet",
'GN':"diamond necklae",
'REP':"jewellery repairing",
'CUF':"diamond cufflink",
'GBT':"gold bracelet with color stone",
'DIA':"gold chain",
'AN':"diamond anklet", 
'GE':"gold earring",
'SUT':"diamond suiti",
'GPN':"gold pendant with colour stone",
'GER':"gold earring",
'RP':"platinum ring",
'GNK':"gold necklace",
'NP':"nose pin", 
'GBNC':'gold bangle with colour stone',
'GHN':"gold hand chain",
'BRCH':"gold brooch",
'GP':"gold pendant",
'JEW':"gold chain",
'GRN':"gold ring with color stone",
'CRN':"diamond crown",
'HC':"hand chain",
'DJEW':"cufflink", 
'BB':"diamond belly button"}

#upload_file=st.sidebar.file_uploader(label="Upload your csv or excel file",type=["csv","xlsx"])
#st_csv_download_button(data)  
#global df
# if upload_file is not None:
#     print(upload_file)
#     print('hello')
#     try:
#         df = pd.read_csv(upload_file)
#     except Exception as e:
#         print(e)
     
    
    
df["Design"]= df.Design.str.lower()
df["Design"]=df["Design"].astype('category')
df=df.replace({'Category_Code':new})
df['Design'] = np.where(df['Design']== "diamond",df["Category_Code"] , df['Design'])
df["QUANTITY"]=1
df2=df[["VOCNO","Design",'QUANTITY',"PRICE_RANGE"]]
#df2 = df2.astype({"PRICE_RANGE": object})

    
basket=df2.groupby(["VOCNO","Design"])["QUANTITY"].sum().unstack().reset_index().fillna(0).set_index("VOCNO")
basket=pd.DataFrame(basket)

def encode_unit(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_set = basket.applymap(encode_unit)

frequent_itemsets = apriori(basket_set, min_support=0.08, use_colnames=True)

def rules_to_coordinates(rules):
    rules['antecedent'] = rules['antecedents'].apply(lambda antecedent: list(antecedent)[0])
    rules['consequent'] = rules['consequents'].apply(lambda consequent: list(consequent)[0])
    rules['rule'] = rules.index
    return rules[['antecedent','consequent','rule']]


from pandas.plotting import parallel_coordinates

#print (frequent_itemsets)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
rules=rules.head(40)
rules= rules.sort_values(by = 'confidence', ascending = False)
#rules.columns = map(str.upper, rules.columns)
rules.columns =["Antecedents","Consequents",'AntecedentSupport','ConsequentSupport',"Support","Confidence","Lift","Leverage","Conviction"]
rules= rules.iloc[:, :-2]
html_table = rules.to_html(justify=CENTER,index=False,classes="table table-bordered dt-responsive",table_id="datatable_wrapper")    
    #rules=rules.style.set_properties(subset=rules.columns, **{'text-align':'left','font-family': 'calibri','font-size': '10px'})
    #rules = rules.style.set_properties(**{
     #       'background-color': 'grey',
      #      'font-size': '10pt',
    #})
    #def colfix(df, L=5):
    #    return df.rename(columns=lambda x: ' '.join(x.replace('_', ' ')[i:i+L] for i in range(0,len(x),L)) if df[x].dtype in ['float64','int64'] else x )

    #colfix(rules)
    #rules.style.set_table_styles([dict(selector="th",props=[('max-width', '30px')])])
    #AgGrid(rules, fit_columns_on_grid_load=True)
    
    # st.markdown(
    # """<style>
    #     .dataframe {text-align: left !important}
    # </style>
    # """, unsafe_allow_html=True) 
    
#frequent_itemsets

# In[16]:

    #st.title("Market Basket Analysis with different Design")
    # new_title = '<p style="font-family:calibri; color:#0094cb; font-size: 36px;">Market Basket Analysis with different Design</p>'
    # st.markdown(new_title, unsafe_allow_html=True)
    # st.write(rules)
    # st.write("----------------------------------------------------------------------------------------")
    
    
    
fig4=pltnew.figure(figsize=(12,20))
#pltnew.rcParams['figure.figsize'] = (10,6)
color = pltnew.cm.inferno(np.linspace(0,1,15))
rules['Antecedents'].value_counts().head(20).plot.bar(color = color)
pltnew.title('Top 20 Most Frequent Items',fontsize=15,color="#0094cb",loc='left')
pltnew.ylabel('Counts')
pltnew.xlabel('Items')
flikes = BytesIO()
pltnew.savefig(flikes)
b641 = base64.b64encode(flikes.getvalue()).decode()
#pltnew.close()
#pltnew.show()
st.pyplot(fig4)

st.write("----------------------------------------------------------------------------------------")


     
from pandas.plotting import parallel_coordinates

# Compute the frequent itemsets
#frequent_itemsets = apriori(onehot, min_support = 0.15, 
#                           use_colnames = True, max_len = 2)

# Compute rules from the frequent itemsets
rules = association_rules(frequent_itemsets, metric = 'confidence', 
                            min_threshold = 0.10)

# Convert rules into coordinates suitable for use in a parallel coordinates plot
coords = rules_to_coordinates(rules.head(40))

# Generate parallel coordinates plot

fig=pltnew.figure(figsize=(12,8))
parallel_coordinates(coords, 'rule')
pltnew.legend([])
pltnew.grid(True)
pltnew.title(' Parallel coordinates to visualize rules', fontsize=10,color="#0094cb",loc='left')
flikes1 = BytesIO()
pltnew.savefig(flikes1)
b642 = base64.b64encode(flikes1.getvalue()).decode()
#st.write("**parallel coordinates to visualize rules**")
st.write(" ")
st.pyplot(fig)
st.write(" ")
st.write(" ")
st.write("----------------------------------------------------------------- ")

fig2=pltnew.figure(figsize=(10,6))
#pltnew.title('Left Title', loc='left')
pltnew.title('Optimality of the support-confidence border ', fontsize=15,color="#0094cb",loc='left')
sns.scatterplot(x = "support", y = "confidence", 
                size = "lift", data = rules)#.set(title="Optimality of the support-confidence border")

pltnew.margins(0.01,0.01)
flikes2 = BytesIO()
pltnew.savefig(flikes2)
b643 = base64.b64encode(flikes2.getvalue()).decode()
#st.write("**Optimality of the support-confidence border**")
st.write(" ")
st.pyplot(fig2)

st.write("An association rule has two parts: an **Antecedent** (if) and a **Consequent** (then). An antecedent is an item found within the data. A consequent is an item found in combination with the antecedent. ... Association rules are calculated from itemsets, which are made up of two or more items.")
#st.write("**Consequent**  : item found in combination with the antecedent")
st.write("**Support**     : Support is an indication of how frequently the items appear in the data. It refers to how often a given rule appears in the database being mined.")
st.write("**Confidence**  : Confidence indicates the number of times the if-then statements are found true.Confidence refers to the amount of times a given rule turns out to be true in practice. A rule may show a strong correlation in a data set because it appears very often but may occur far less when applied. This would be a case of high support, but low confidence.Conversely, a rule might not particularly stand out in a data set, but continued analysis shows that it occurs very frequently. This would be a case of high confidence and low support. Using these measures helps analysts separate causation from correlation, and allows them to properly value a given rule. ")
st.write("**Lift**        : lift can be used to compare confidence with expected confidence, or how many times an if-then statement is expected to be found true. It is the ratio of confidence to support. If the lift value is a negative value, then there is a negative correlation between datapoints. If the value is positive, there is a positive correlation, and if the ratio equals 1, then there is no correlation.")


st.write("----------------------------------------------------------------- ")
df2 = df2.astype({"PRICE_RANGE": object})
basket=df2.groupby(["VOCNO","PRICE_RANGE"])["QUANTITY"].sum().unstack().reset_index().fillna(0).set_index("VOCNO")
basket=pd.DataFrame(basket)

def encode_unit(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_set = basket.applymap(encode_unit)

frequent_itemsets = apriori(basket_set, min_support=0.08, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
rules=rules.head(40)
rules= rules.sort_values(by = 'confidence', ascending = False)
#rules.columns = map(str.upper, rules.columns)
rules.columns =["Antecedents","Consequents",'AntecedentSupport','ConsequentSupport',"Support","Confidence","Lift","Leverage","Conviction"]
rules= rules.iloc[:, :-2]

st.markdown(
"""<style>
    .dataframe {text-align: left !important}
</style>
""", unsafe_allow_html=True) 

#frequent_itemsets

# In[16]:

#st.title("Market Basket Analysis with different Price Range")
new_title2 = '<p style="font-family:calibri; color:#0094cb; font-size: 36px;">Market Basket Analysis with different Price Range</p>'
st.markdown(new_title2, unsafe_allow_html=True)
st.write(rules)
st.write(" ")
st.write(" ")

st.write("----------------------------------------------------------------------------------------")

fig3=pltnew.figure(figsize=(10,18))
#pltnew.rcParams['figure.figsize'] = (10,6)
color = pltnew.cm.inferno(np.linspace(0,1,20))
rules['Antecedents'].value_counts().head(20).plot.bar(color = color)
pltnew.title('Top 20 Most Frequent Items',fontsize=15,color="#0094cb",loc='left')
pltnew.ylabel('Counts')
pltnew.xlabel('Items')
flikes3 = BytesIO()
pltnew.savefig(flikes3)
b644 = base64.b64encode(flikes3.getvalue()).decode()
#pltnew.show()
st.pyplot(fig3)

st.write("----------------------------------------------------------------------------------------")


rules = association_rules(frequent_itemsets, metric = 'confidence', 
                            min_threshold = 0.55)
html_table_2 = rules.to_html(justify=CENTER,index=False,classes="table table-bordered dt-responsive",table_id="datatable_wrapper_2") 
# Convert rules into coordinates suitable for use in a parallel coordinates plot
coords = rules_to_coordinates(rules.head(40))

# Generate parallel coordinates plot

fig=pltnew.figure(figsize=(10,8))
parallel_coordinates(coords, 'rule')
pltnew.legend([])
pltnew.grid(True)
pltnew.title(' Parallel coordinates to visualize rules', fontsize=15,color="#0094cb",loc='left')
flikes4 = BytesIO()
pltnew.savefig(flikes4)
b645 = base64.b64encode(flikes4.getvalue()).decode()
#st.write("**parallel coordinates to visualize rules**")
st.write(" ")
st.pyplot(fig)




# Create your views here.
class EmailInbox(LoginRequiredMixin,TemplateView):
    template_name = "email/email-inbox.html"
class EmailRead(LoginRequiredMixin,TemplateView):
    template_name = "email/email-read.html"
class EmailCompose(LoginRequiredMixin,TemplateView):
    template_name = "email/email-compose.html"

def testapi(request):
    return render(request, template_name='pages/utility/chart2.html', context={'wind_rose': b641,'chart2':b642,'chart3':b643,'chart4':b644,'chart5':b645,'tablesdata':html_table,'tablesdata2':html_table_2}) 
     
