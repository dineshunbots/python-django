from django.shortcuts import render,HttpResponse
from django.views.generic.base import TemplateView
from django.http.response import JsonResponse
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.decorators.csrf import csrf_exempt
from django.contrib.sessions.models import Session
from http import cookies
import numpy as np
import pandas as pd
from apyori import apriori
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from mlxtend.frequent_patterns import apriori, association_rules


df=pd.read_csv("Market basket analysis data.csv",on_bad_lines='skip')
#df.Number
df.tail(30)



# In[16]:


df['PRICE_RANGE'] = pd.cut(x=df["NETVALUECC1"], bins=[0,2000,10000,20000,30000,40000,50000,60000,75000,600000],labels=['0-2000', '2000-10000', '10000-20000',"20000-30000","30000-40000","40000-50000","50000-60000","60000-75000","75000 and above"])
#pd.cut(df['some_col'], bins=[0,20,40,60], labels=['0-20', '20-40', '40-60']) 
df=df.dropna()


# In[17]:


df


# In[18]:


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


# In[19]:



df.Design= df.Design.str.lower()
df["Design"]=df["Design"].astype('category')
#df["PRICE_RANGE"]=df["PRICE_RANGE"].astype('category')
df=df.replace({'Category_Code':new})
df['Design'] = np.where(df['Design']== "diamond",df["Category_Code"] , df['Design'])
df["QUANTITY"]=1
#df['VOCNO'] = pd.Categorical(df['VOCNO']
#df['VOCNO'] = df['VOCNO'].cat.add_categories('Unknown')
df2=df[["VOCNO","Design","PRICE_RANGE",'QUANTITY',"GROSS_WT1"]]

#df2['VOCNO'] = pd.Categorical(df['VOCNO'],categories=df['VOCNO'].dropna().unique())
#df2.to_csv("file one.csv")
df2 = df2.astype({"PRICE_RANGE": object})


# In[36]:



basket=df2.groupby(["VOCNO","PRICE_RANGE"])["GROSS_WT1"].sum().unstack().reset_index().fillna(0).set_index("VOCNO")
basket=pd.DataFrame(basket)
basket=basket.head(200)
#basket.to_csv('file7.csv')


# In[37]:


def encode_unit(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
    
basket_set = basket.applymap(encode_unit)
#basket_set.to_csv('file4.csv')
basket


# In[38]:


frequent_itemsets = apriori(basket_set, min_support=0.08, use_colnames=True)
#print (frequent_itemsets)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
rules


# In[23]:


support=rules['support'].values
confidence=rules['confidence'].values
import random


# In[24]:


for i in range (len(support)):
   support[i] = support[i] + 0.0025 * (random.randint(1,10) - 5) 
   confidence[i] = confidence[i] + 0.0025 * (random.randint(1,10) - 5)
 
# plt.scatter(support, confidence,   alpha=0.5, marker="o")
# plt.xlabel('support')
# plt.ylabel('confidence') 
# plt.show()


# In[25]:


rules


# In[26]:



rules.columns = map(str.upper, rules.columns)

rules


# In[27]:




rules["SUPPORT"] = pd.Series([round(val, 2) for val in rules["SUPPORT"]], index = rules.index)
rules["SUPPORT"]=rules["SUPPORT"]*100


# In[29]:


def rules_to_coordinates(rules):
    rules['antecedent'] = rules['antecedents'].apply(lambda antecedent: list(antecedent)[0])
    rules['consequent'] = rules['consequents'].apply(lambda consequent: list(consequent)[0])
    rules['rule'] = rules.index
    return rules[['antecedent','consequent','rule']]


# In[208]:


from pandas.plotting import parallel_coordinates

# Compute the frequent itemsets
# frequent_itemsets = apriori(onehot, min_support = 0.15, 
#                            use_colnames = True, max_len = 2)

# Compute rules from the frequent itemsets
rules = association_rules(frequent_itemsets, metric = 'confidence', 
                          min_threshold = 0.55)

# Convert rules into coordinates suitable for use in a parallel coordinates plot
coords = rules_to_coordinates(rules.head(40))

# Generate parallel coordinates plot
# plt.figure(figsize=(4,8))
# parallel_coordinates(coords, 'rule')
# plt.legend([])
# plt.grid(True)
# plt.show()


# In[30]:


rules = association_rules(frequent_itemsets, metric='support', 
                          min_threshold = 0.0)

# Generate scatterplot using support and confidence
# plt.figure(figsize=(10,6))
# sns.scatterplot(x = "support", y = "confidence", data = rules)
# plt.margins(0.01,0.01)
# plt.show()


# In[31]:


# plt.figure(figsize=(10,6))
# sns.scatterplot(x = "support", y = "confidence", 
#                 size = "lift", data = rules)
# plt.margins(0.01,0.01)
# plt.show()


# In[32]:


import plotly.express as px
from pandas.plotting import parallel_coordinates


    # Compute the frequent itemsets
    #frequent_itemsets = apriori(onehot, min_support = 0.15, 
    #                           use_colnames = True, max_len = 2)

    # Compute rules from the frequent itemsets
#rules = association_rules(frequent_itemsets, metric = 'confidence', 
#                          min_threshold = 0.55)

    # Convert rules into coordinates suitable for use in a parallel coordinates plot
#coords = rules_to_coordinates(rules.head(40))

    # Generate parallel coordinates plot
    
#fig=px.parallel_coordinates(rules, ['antecedents', 'consequents'] )
#parallel_coordinates(coords, 'rule')
#px.legend([])
#px.grid(True)
#px.title(' parallel coordinates to visualize rules', fontsize=15,color="blue")
#fig.show()


# In[ ]:


rules.columns =["antecedents","consequents",'antecedentSupport','consequentSupport',"support","confidence","lift","leverage","conviction"]


# In[217]:


rules


# In[213]:


import plotly.express as px
# df = px.data.iris()
# fig = px.parallel_coordinates(rules.head(20), color="confidence", labels={"antecedentSupport": "antecedentSupport",
#                 "consequentSupport": "consequentSupport", "support": "support",
#                 "lift":"lift", "levarage":"levarage", "conviction": "conviction" },
#                              color_continuous_scale=px.colors.diverging.Tealrose,
#                              color_continuous_midpoint=1)
# fig.show()
#antecedent,support,consequent,support,support,confidence,lift,leverage,conviction


# In[218]:


# plt.rcParams['figure.figsize'] = (10,6)
# color = plt.cm.inferno(np.linspace(0,1,20))
# rules['antecedents'].value_counts().head(20).plot.bar(color = color)
# plt.title('Top 20 Most Frequent Items')
# plt.ylabel('Counts')
# plt.xlabel('Items')
#plt.show()

# Authentication
class AuthLockScrren(LoginRequiredMixin,TemplateView):
    template_name = "pages/authentication/auth-lock-screen.html"
class AuthLogin(LoginRequiredMixin,TemplateView):
    template_name = "pages/authentication/auth-login.html"
class AuthRegister(LoginRequiredMixin,TemplateView):
    template_name = "pages/authentication/auth-register.html"
class AuthRecoverpw(LoginRequiredMixin,TemplateView):
    template_name = "pages/authentication/auth-recoverpw.html"

# Pages
class Error404(LoginRequiredMixin,TemplateView):
    template_name = "pages/utility/pages-404.html"
class Error500(LoginRequiredMixin,TemplateView):
    template_name = "pages/utility/pages-500.html"
class ComingSoon(LoginRequiredMixin,TemplateView):
    template_name = "pages/utility/pages-comingsoon.html"
class Faqs(LoginRequiredMixin,TemplateView):
    template_name = "pages/utility/pages-faqs.html"
class Maintenance(LoginRequiredMixin,TemplateView):
    template_name = "pages/utility/pages-maintenance.html"
class Pricing(LoginRequiredMixin,TemplateView):
    template_name = "pages/utility/pages-pricing.html"   
class Starter(LoginRequiredMixin,TemplateView):
    tag = "hi"
    #return('pages/utility/pages-starter.html', context=tag)
    template_name = "pages/utility/pages-starter.html"
    # template_name = "pages/utility/MBA_APRIORI.py"
class Timeline(LoginRequiredMixin,TemplateView):
    template_name = "pages/utility/pages-timeline.html"

def chart1(request):
    plt.scatter(support, confidence,   alpha=0.5, marker="o")
    plt.xlabel('support')
    plt.ylabel('confidence') 
    flike = io.BytesIO()
    plt.savefig(flike)
    b64 = base64.b64encode(flike.getvalue()).decode()
    plt.close()
    return render(request, template_name='pages/utility/chart1.html', context={'wind_rose': b64})

def chart2(request):
    plt.figure(figsize=(8,8))
    parallel_coordinates(coords, 'rule')
    plt.legend([])
    plt.grid(True)
    flike = io.BytesIO()
    plt.savefig(flike)
    b64 = base64.b64encode(flike.getvalue()).decode()
    plt.close()
    return render(request, template_name='pages/utility/chart2.html', context={'wind_rose': b64})   

def chart3(request):
    plt.figure(figsize=(10,6))
    sns.scatterplot(x = "support", y = "confidence", data = rules)
    plt.margins(0.01,0.01)
    flike = io.BytesIO()
    plt.savefig(flike)
    b64 = base64.b64encode(flike.getvalue()).decode()
    plt.close()
    return render(request, template_name='pages/utility/chart3.html', context={'wind_rose': b64})  

def chart4(request):
    plt.figure(figsize=(10,6))
    sns.scatterplot(x = "support", y = "confidence", size = "lift", data = rules)
    plt.margins(0.01,0.01)
    flike = io.BytesIO()
    plt.savefig(flike)
    b64 = base64.b64encode(flike.getvalue()).decode()
    plt.close()
    return render(request, template_name='pages/utility/chart4.html', context={'wind_rose': b64})   
    
def chart5(request):
    
    plt.rcParams['figure.figsize'] = (10,6)
    color = plt.cm.inferno(np.linspace(0,1,20))
    rules['antecedents'].value_counts().head(20).plot.bar(color = color)
    plt.title('Top 20 Most Frequent Items')
    plt.ylabel('Counts')
    plt.xlabel('Items')
    flike = io.BytesIO()
    plt.savefig(flike)
    b64 = base64.b64encode(flike.getvalue()).decode()
    plt.close()
    return render(request, template_name='pages/utility/chart5.html', context={'wind_rose': b64}) 

def chart6(request):
    
    rules = association_rules(frequent_itemsets, metric = 'confidence',  min_threshold = 0.55)
    coords = rules_to_coordinates(rules.head(40))  
   
    #fig=px.parallel_coordinates(rules, ['antecedents', 'consequents'] )
    parallel_coordinates(coords, 'rule')
    plt.legend([])
    plt.grid(True)
    plt.title(' parallel coordinates to visualize rules', fontsize=15,color="blue")
    #fig.show()
    flike = io.BytesIO()
    plt.savefig(flike)
    b64 = base64.b64encode(flike.getvalue()).decode()
    plt.close()
    return render(request, template_name='pages/utility/chart6.html', context={'wind_rose': b64}) 

def chart7(request):
    df = px.data.iris()
    fig = px.parallel_coordinates(rules.head(20), color="confidence", labels={"antecedentSupport": "antecedentSupport",
                "consequentSupport": "consequentSupport", "support": "support",
                "lift":"lift", "levarage":"levarage", "conviction": "conviction" },
                             color_continuous_scale=px.colors.diverging.Tealrose,
                             color_continuous_midpoint=1)
    #fig.show()
    flike = io.BytesIO()
    px.savefig(flike)
    b64 = base64.b64encode(flike.getvalue()).decode()
    return render(request, template_name='pages/utility/chart7.html', context={'wind_rose': b64}) 

def dashboard(request):
    return render(request, template_name='pages/utility/dashboard.html', context={})

def viewrecommendation(request):
    return render(request, template_name='pages/utility/viewrecommendation.html', context={})

def dashboardlogin(request): 
        return render(request, template_name='pages/utility/login.html', context={})
  

@csrf_exempt        
def login(request): 
    email =     request.POST['email']
    password =     request.POST['password']
    result = ""
    responses = {
                    "Status": result,
                    
    }
    return JsonResponse(responses)
def marketbasketanalysis(request):
    plt.scatter(support, confidence,   alpha=0.5, marker="o")
    plt.xlabel('support')
    plt.ylabel('confidence') 
    flike1 = io.BytesIO()
    plt.savefig(flike1)
    b641 = base64.b64encode(flike1.getvalue()).decode()
    plt.close()
    plt.figure(figsize=(8,8))
    coords = rules_to_coordinates(rules.head(40))
    parallel_coordinates(coords, 'rule')
    plt.legend([])
    plt.grid(True)
    flike2 = io.BytesIO()
    plt.savefig(flike2)
    b642 = base64.b64encode(flike2.getvalue()).decode()
    plt.close()
    plt.figure(figsize=(10,6))
    sns.scatterplot(x = "support", y = "confidence", data = rules)
    plt.margins(0.01,0.01)
    flike3 = io.BytesIO()
    plt.savefig(flike3)
    b643 = base64.b64encode(flike3.getvalue()).decode()
    plt.close()
    plt.figure(figsize=(10,6))
    sns.scatterplot(x = "support", y = "confidence", size = "lift", data = rules)
    plt.margins(0.01,0.01)
    flike4 = io.BytesIO()
    plt.savefig(flike4)
    b644 = base64.b64encode(flike4.getvalue()).decode()
    plt.close()
    plt.rcParams['figure.figsize'] = (10,6)
    color = plt.cm.inferno(np.linspace(0,1,20))
    rules['antecedents'].value_counts().head(20).plot.bar(color = color)
    plt.title('Top 20 Most Frequent Items')
    plt.ylabel('Counts')
    plt.xlabel('Items')
    flike5 = io.BytesIO()
    plt.savefig(flike5)
    b645 = base64.b64encode(flike5.getvalue()).decode()
    plt.close()
    
    #rules = association_rules(frequent_itemsets, metric = 'confidence',  min_threshold = 0.55)
    coords = rules_to_coordinates(rules.head(40))  
    #fig=px.parallel_coordinates(rules, ['antecedents', 'consequents'] )
    parallel_coordinates(coords, 'rule')
    plt.legend([])
    plt.grid(True)
    plt.title(' parallel coordinates to visualize rules', fontsize=15,color="blue")
    #fig.show()
    flike6 = io.BytesIO()
    plt.savefig(flike6)
    b646 = base64.b64encode(flike6.getvalue()).decode()
    plt.close()
    return render(request, template_name='pages/utility/marketbasketanalysis.html', context={'wind_rose1': b641,'wind_rose2': b642,'wind_rose3': b643,'wind_rose4': b644,'wind_rose5': b645,'wind_rose6': b646})
