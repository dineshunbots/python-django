from tkinter import CENTER
from django.shortcuts import render
from django.views.generic.base import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin

import numpy as np
import pandas as pd
from apyori import apriori
import base64
import io
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
from datetime import datetime
from mlxtend.frequent_patterns import apriori, association_rules

import requests
import json



## API CALLING

res = requests.get("http://brilliantbidata.sunwebapps.com/api/MarketBasket?strFromDate=2022/01/01&strTodate=today")
j = res.json()
df = pd.DataFrame(j)
df.head(50)
df=df[df["VOCTYPE"].isin(["PS1","PSR"])]
df
df['NETVALUECC1'] = df['NETVALUECC1'].astype(float)
df['PRICE_RANGE'] = pd.cut(x=df["NETVALUECC1"], bins=[0,2000,10000,20000,30000,40000,50000,60000,75000,600000],labels=['0-2000', '2000-10000', '10000-20000',"20000-30000","30000-40000","40000-50000","50000-60000","60000-75000","75000 and above"])
#pd.cut(df['some_col'], bins=[0,20,40,60], labels=['0-20', '20-40', '40-60']) 
df=df.dropna()
df
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
df.DESIGN_DESCRIPTION= df.DESIGN_DESCRIPTION.str.lower()
df["Design"]=df["DESIGN_DESCRIPTION"].astype('category')
#df["PRICE_RANGE"]=df["PRICE_RANGE"].astype('category')
df=df.replace({'CATEGORY_CODE':new})
df['Design'] = np.where(df['Design']== "diamond",df["CATEGORY_CODE"] , df['Design'])
df["QUANTITY"]=1
#df['VOCNO'] = pd.Categorical(df['VOCNO']
#df['VOCNO'] = df['VOCNO'].cat.add_categories('Unknown')
df2=df[["VOCNO","Design","PRICE_RANGE",'QUANTITY',"GROSS_WT1"]]
df['VOCNO'] = df['VOCNO'].astype(float)
df['GROSS_WT1'] = df['GROSS_WT1'].astype(float)
df['GROSS_WT1'] = df['GROSS_WT1']*-1

#df2=df[["VOCNO","Design",'QUANTITY',"GROSS_WT1"]]

#df2['VOCNO'] = pd.Categorical(df['VOCNO'],categories=df['VOCNO'].dropna().unique())
#df2.to_csv("file one.csv")
df2 = df2.astype({"PRICE_RANGE": object})
basket=df2.groupby(["VOCNO","PRICE_RANGE"])["GROSS_WT1"].sum().unstack().reset_index().fillna(0).set_index("VOCNO")
#basket=df2.groupby(["VOCNO","Design"])["GROSS_WT1"].sum().unstack().reset_index().fillna(0).set_index("VOCNO")
basket=pd.DataFrame(basket) 
basket=basket.head(200)
#basket.to_csv('file7.csv')
def encode_unit(x):
    if str(x) == "":
        return 0 
    if str(x) != 1:
        return 1
    
basket_set = basket.applymap(encode_unit)
#basket_set.to_csv('file4.csv')
basket_set.dropna()
basket_set.isnull().values.sum()
frequent_itemsets = apriori(basket_set, min_support=0.08, use_colnames=True)
#print (frequent_itemsets)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
rules
support=rules['support'].values
confidence=rules['confidence'].values
import random
for i in range (len(support)):
   support[i] = support[i] + 0.0025 * (random.randint(1,10) - 5) 
   confidence[i] = confidence[i] + 0.0025 * (random.randint(1,10) - 5)
 












# Vertical
class Boxed(LoginRequiredMixin,TemplateView):
    template_name = "layout/vertical/layouts-boxed.html"
class CompactSidebar(LoginRequiredMixin,TemplateView):
    template_name = "layout/vertical/layouts-compact-sidebar.html"
class IconSidebar(LoginRequiredMixin,TemplateView):
    template_name = "layout/vertical/layouts-icon-sidebar.html"
class Lightsidebar(LoginRequiredMixin,TemplateView):
    template_name = "layout/vertical/layouts-light-sidebar.html"


# Vertical
class Horizontal(LoginRequiredMixin,TemplateView):
    template_name = "layout/horizontal/layouts-horizontal.html"
class HoriBoxedWidth(LoginRequiredMixin,TemplateView):
    template_name = "layout/horizontal/layouts-hori-boxed-width.html"
class HoriTopbardark (LoginRequiredMixin,TemplateView):
    template_name = "layout/horizontal/layouts-hori-topbar-dark.html"


def marketbasketanalysisapi(request):

    plt.scatter(support, confidence,   alpha=0.5, marker="o")
    plt.xlabel('support')
    plt.ylabel('confidence') 
    #plt.show()
    flike = io.BytesIO()
    
    plt.savefig(flike)
    b64 = base64.b64encode(flike.getvalue()).decode()
    plt.close() 
    html_table = rules.to_html(justify=CENTER,index=False,classes="table table-bordered dt-responsive",table_id="datatable_wrapper")
    return render(request, template_name='pages/utility/chart1.html', context={'wind_rose': b64,'tablesdata':html_table})
