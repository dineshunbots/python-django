from tkinter import CENTER
from django.shortcuts import render
from django.views.generic.base import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http.response import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import base64
import io

import numpy as np
import pandas as pd
from apyori import apriori
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from mlxtend.frequent_patterns import apriori, association_rules
import requests
import json

# import numpy as np
# import pandas as pd
# from apyori import apriori
# import base64
# import io
# import matplotlib.pyplot as plt
# plt.switch_backend('agg')
# import seaborn as sns
# from datetime import datetime
# from mlxtend.frequent_patterns import apriori, association_rules

# import requests
# import json



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
 

def rules_to_coordinates(rules):
    rules['antecedent'] = rules['antecedents'].apply(lambda antecedent: list(antecedent)[0])
    rules['consequent'] = rules['consequents'].apply(lambda consequent: list(consequent)[0])
    rules['rule'] = rules.index
    return rules[['antecedent','consequent','rule']]










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

def marketbasketanalysischart(request):
    return render(request, template_name='pages/utility/chart3.html', context={'wind_rose': "",})

@csrf_exempt        
def getapirecord(request): 
    fdate =     request.POST['fdate']
    tdate =     request.POST['tdate']
    res = requests.get("http://brilliantbidata.sunwebapps.com/api/MarketBasket?strFromDate="+fdate+"&strTodate="+tdate)
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

    fig=plt.figure(figsize=(6,5))        
    plt.scatter(support, confidence,   alpha=0.5, marker="o")
    plt.xlabel('support')
    plt.ylabel('confidence') 
    #plt.show()
    flike1 = io.BytesIO()
    
    plt.savefig(flike1)
    b641 = base64.b64encode(flike1.getvalue()).decode()
    plt.close() 



    from pandas.plotting import parallel_coordinates

    rules = association_rules(frequent_itemsets, metric = 'confidence', 
                            min_threshold = 0.55)

    # Convert rules into coordinates suitable for use in a parallel coordinates plot
    coords = rules_to_coordinates(rules.head(40))

    # Generate parallel coordinates plot
    plt.figure(figsize=(6,5))
    parallel_coordinates(coords, 'rule')
    plt.legend([])
    plt.grid(True)
    flike2 = io.BytesIO()
    plt.savefig(flike2,bbox_inches='tight')
    plt.tight_layout()
    plt.savefig(flike2)
    b642 = base64.b64encode(flike2.getvalue()).decode()
    plt.close() 
    #plt.show()

    plt.rcParams['figure.figsize'] = (6,5)
    color = plt.cm.inferno(np.linspace(0,1,20))
    rules['antecedents'].value_counts().head(20).plot.bar(color = color)
    plt.title('Top 20 Most Frequent Items')
    plt.ylabel('Counts')
    plt.xlabel('Items')
    flike3 = io.BytesIO()
    plt.savefig(flike3,bbox_inches='tight')
    plt.tight_layout()
    plt.savefig(flike3)
    b643 = base64.b64encode(flike3.getvalue()).decode()
    plt.close() 
    #plt.show()

    # import networkx as nx
    # df['diamond'] = 'diamond ring'
    # diamond = df.truncate(before = -1, after = 70)
    # diamond = nx.from_pandas_edgelist(diamond, source = 'diamond', target = 'Design', edge_attr = True)

    # import warnings
    # warnings.filterwarnings('ignore')

    # plt.rcParams['figure.figsize'] = (13, 11)
    # pos = nx.spring_layout(diamond)
    # color = plt.cm.Set1(np.linspace(0, 40, 1))
    # nx.draw_networkx_nodes(diamond, pos, node_size = 12000, node_color = color)
    # nx.draw_networkx_edges(diamond, pos, width = 2, alpha = 0.6, edge_color = 'black')
    # nx.draw_networkx_labels(diamond, pos, font_size = 12, font_family = 'sans-serif')
    # plt.axis('off')
    # plt.grid()
    # plt.title('Top 15 First Choices', fontsize = 20)
    # #pltnew.show()
    # flikes6 = io.BytesIO()
    # plt.savefig(flikes6,bbox_inches='tight')
    # plt.tight_layout()
    # b647 = base64.b64encode(flikes6.getvalue()).decode()
    # plt.close()


    ress = requests.get("http://brilliantbidata.sunwebapps.com/api/MarketBasket?strFromDate="+fdate+"&strTodate="+tdate)
    jes = ress.json()
    dfs = pd.DataFrame(jes)
    dfs
    dfs.DESIGN_DESCRIPTION= dfs.DESIGN_DESCRIPTION.str.lower()
    dfs["DESIGN_DESCRIPTION"]=dfs["DESIGN_DESCRIPTION"].astype('category')
    dfs["QUANTITY"]=1
    df21=dfs[["VOCNO","DESIGN_DESCRIPTION",'QUANTITY']]
    df21
    #print(df21)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    
    basket1=df21.groupby(["VOCNO","DESIGN_DESCRIPTION"])["QUANTITY"].sum().unstack().reset_index().fillna(0).set_index("VOCNO")
    basket1=pd.DataFrame(basket1)
    basket1.head(300)
    
    basket_set1 = basket1.applymap(encode_unit)
    basket_set1
   
    import networkx as nx
    dfs['diamond'] = "diamond ring"
    diamond = dfs.truncate(before = -1, after = 70) 
    diamond = nx.from_pandas_edgelist(diamond, source = 'diamond', target = 'DESIGN_DESCRIPTION', edge_attr = True)

    import warnings
    warnings.filterwarnings('ignore')

    plt.rcParams['figure.figsize'] = (13, 11)
    pos = nx.spring_layout(diamond)
    color = plt.cm.Set1(np.linspace(0, 40, 1))
    nx.draw_networkx_nodes(diamond, pos, node_size = 12000, node_color = color)
    nx.draw_networkx_edges(diamond, pos, width = 2, alpha = 0.6, edge_color = 'black')
    nx.draw_networkx_labels(diamond, pos, font_size = 12, font_family = 'sans-serif')
    plt.axis('off')
    plt.grid()
    plt.title('Top 15 First Choices', fontsize = 20)
    #pltnew.show()
    flikes6new = io.BytesIO()
    plt.savefig(flikes6new,bbox_inches='tight')
    plt.tight_layout()
    b647 = base64.b64encode(flikes6new.getvalue()).decode()
    plt.close()



    html_tables = rules.to_html(justify=CENTER,index=False,classes="table table-bordered dt-responsive",table_id="datatable_wrapper_3")    
    result = "SUCCESS"
    responses = {
                    "Status": result,
                    "res":b641,
                    "chart1":b642,
                    "chart2":b643,
                    "chart3":b647,
                    "table":html_tables,
                    
    }
    return JsonResponse(responses)