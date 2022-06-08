from django.shortcuts import render
from django.views.generic.base import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http.response import JsonResponse
from django.views.decorators.csrf import csrf_exempt


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


import requests
import json

# UI-Elements
class Alerts(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-alerts.html"
class Badge(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-badge.html"
class Breadcrumb(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-breadcrumb.html"
class Buttons(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-buttons.html"
class Cards(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-cards.html"
class Carousel(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-carousel.html"
class Dropdowns(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-dropdowns.html"
class Grid(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-grid.html"
class Images(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-images.html"
class Lightbox(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-lightbox.html"
class Modals(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-modals.html"
class Offcanvas(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-offcanvas.html"
class Pagination(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-pagination.html"
class Placeholders(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-placeholders.html"
class PopoverTooltips(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-popover-tooltips.html"
class Progressbars(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-progressbars.html"
class Rangeslider(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-rangeslider.html"
class Rating(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-rating.html"
class SessionTimeout(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-session-timeout.html"
class SweetAlert(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-sweet-alert.html"
class TabsAccordions(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-tabs-accordions.html"
class Toasts(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-toasts.html"
class Typography(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-typography.html"
class Video(LoginRequiredMixin,TemplateView):
    template_name = "components/ui-elements/ui-video.html"


# Forms
class Advanced(LoginRequiredMixin,TemplateView):
    template_name = "components/forms/form-advanced.html"
class Editors(LoginRequiredMixin,TemplateView):
    template_name = "components/forms/form-editors.html"
class Elements(LoginRequiredMixin,TemplateView):
    template_name = "components/forms/form-elements.html"
class Mask(LoginRequiredMixin,TemplateView):
    template_name = "components/forms/form-mask.html"
class Uploads(LoginRequiredMixin,TemplateView):
    template_name = "components/forms/form-uploads.html"
class Validation(LoginRequiredMixin,TemplateView):
    template_name = "components/forms/form-validation.html"
class Wizard(LoginRequiredMixin,TemplateView):
    template_name = "components/forms/form-wizard.html"
class Xeditable(LoginRequiredMixin,TemplateView):
    template_name = "components/forms/form-xeditable.html"

# Tables
class Basic(LoginRequiredMixin,TemplateView):
    template_name = "components/tables/tables-basic.html"
class Datatable(LoginRequiredMixin,TemplateView):
    template_name = "components/tables/tables-datatable.html"
class Editable(LoginRequiredMixin,TemplateView):
    template_name = "components/tables/tables-editable.html"
class Responsive(LoginRequiredMixin,TemplateView):
    template_name = "components/tables/tables-responsive.html"

# Charts
class Apex(LoginRequiredMixin,TemplateView):
    template_name = "components/charts/charts-apex.html"
class Chartjs(LoginRequiredMixin,TemplateView):
    template_name = "components/charts/charts-chartjs.html"
class Flot(LoginRequiredMixin,TemplateView):
    template_name = "components/charts/charts-flot.html"
class Knob(LoginRequiredMixin,TemplateView):
    template_name = "components/charts/charts-knob.html"
class Sparkline(LoginRequiredMixin,TemplateView):
    template_name = "components/charts/charts-sparkline.html"


# Icons
class Dripicons(LoginRequiredMixin,TemplateView):
    template_name = "components/icons/icons-dripicons.html"
class Fontawesome(LoginRequiredMixin,TemplateView):
    template_name = "components/icons/icons-fontawesome.html"
class Materialdesign(LoginRequiredMixin,TemplateView):
    template_name = "components/icons/icons-materialdesign.html"
class Remix(LoginRequiredMixin,TemplateView):
    template_name = "components/icons/icons-remix.html"

# Maps
class Google(LoginRequiredMixin,TemplateView):
    template_name = "components/maps/maps-google.html"
class Vector(LoginRequiredMixin,TemplateView):
    template_name = "components/maps/maps-vector.html"
def marketbasket(request):
    return render(request, template_name='pages/utility/chart2new.html') 

@csrf_exempt        
def savefile(request):   
    handle_uploaded_file(request.FILES['file'])  
    name = "files/"+request.FILES['file'].name
    df=pd.read_csv(name,on_bad_lines='skip')
    #df.Number
    df.tail(30)

    df['PRICE_RANGE'] = pd.cut(x=df["NETVALUECC1"], bins=[0,2000,10000,20000,30000,40000,50000,60000,75000,600000],labels=['0-2K', '2k-10k', '10k-20k',"20k-30k","30k-40k","40k-50k","50k-60k","60k-75k","75k and above"])
        #pd.cut(df['some_col'], bins=[0,20,40,60], labels=['0-20', '20-40', '40-60']) 
    df=df.dropna()

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

    
        
    df["Design"]= df.Design.str.lower()
    df["Design"]=df["Design"].astype('category')
    df=df.replace({'Category_Code':new})
    df['Design'] = np.where(df['Design']== "diamond",df["Category_Code"] , df['Design'])
    df['Design'] = np.where(df['Design']== "",df["Category_Code"] , df['Design'])
    dess = df['Design'].unique()
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
        
        
    fig4=pltnew.figure(figsize=(10,5))
    #pltnew.rcParams['figure.figsize'] = (10,6)
    color = pltnew.cm.inferno(np.linspace(0,1,15))
    rules['Antecedents'].value_counts().head(20).plot.bar(color = color)
    pltnew.title('Top 20 Most Frequent Items',fontsize=15,color="#0094cb",loc='left')
    pltnew.ylabel('Counts')
    pltnew.xlabel('Items')
    flikes = BytesIO()
    pltnew.savefig(flikes,bbox_inches='tight')
    pltnew.tight_layout()
    b641 = base64.b64encode(flikes.getvalue()).decode()
    pltnew.close()
    #pltnew.close()
    #pltnew.show()
    #st.pyplot(fig4)

    st.write("----------------------------------------------------------------------------------------")


        
    from pandas.plotting import parallel_coordinates

  
    rules = association_rules(frequent_itemsets, metric = 'confidence', 
                                min_threshold = 0.10)

    # Convert rules into coordinates suitable for use in a parallel coordinates plot
    coords = rules_to_coordinates(rules.head(40))

    # Generate parallel coordinates plot

    fig=pltnew.figure(figsize=(10,5))
    parallel_coordinates(coords, 'rule')
    pltnew.legend([])
    pltnew.grid(True)
    pltnew.title(' Parallel Coordinates to Visualize Rules', fontsize=15,color="#0094cb",loc='left')
    # pltnew.ylabel('Antecedents')
    # pltnew.xlabel('Consequents')
    flikes1 = BytesIO()
    pltnew.savefig(flikes1,bbox_inches='tight')
    pltnew.tight_layout()
    b642 = base64.b64encode(flikes1.getvalue()).decode()
    pltnew.close()
    #st.write("**parallel coordinates to visualize rules**")
    st.write(" ")
    st.pyplot(fig)
    st.write(" ")
    st.write(" ")
    st.write("----------------------------------------------------------------- ")

    fig2=pltnew.figure(figsize=(10,5))
    #pltnew.title('Left Title', loc='left')
    pltnew.title('Optimality of the Support-Confidence Border ', fontsize=15,color="#0094cb",loc='left')
    sns.scatterplot(x = "support", y = "confidence", 
                    size = "lift", data = rules)#.set(title="Optimality of the support-confidence border")

    pltnew.margins(0.01,0.01)
    flikes2 = BytesIO()
    pltnew.savefig(flikes2,bbox_inches='tight')
    pltnew.tight_layout()
    b643 = base64.b64encode(flikes2.getvalue()).decode()
    pltnew.close()
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

   
    #st.title("Market Basket Analysis with different Price Range")
    new_title2 = '<p style="font-family:calibri; color:#0094cb; font-size: 36px;">Market Basket Analysis with different Price Range</p>'
    st.markdown(new_title2, unsafe_allow_html=True)
    st.write(rules)
    st.write(" ")
    st.write(" ")

    st.write("----------------------------------------------------------------------------------------")

    fig3=pltnew.figure(figsize=(10,5))
    #pltnew.rcParams['figure.figsize'] = (10,6)
    color = pltnew.cm.inferno(np.linspace(0,1,20))
    rules['Antecedents'].value_counts().head(20).plot.bar(color = color)
    pltnew.title('Top 20 Most Frequent Items',fontsize=15,color="#0094cb",loc='left')
    pltnew.ylabel('Counts')
    pltnew.xlabel('Items')
    flikes3 = BytesIO()
    pltnew.savefig(flikes3,bbox_inches='tight')
    pltnew.tight_layout()
    b644 = base64.b64encode(flikes3.getvalue()).decode()
    pltnew.close()
    #pltnew.show()
    st.pyplot(fig3)

    st.write("----------------------------------------------------------------------------------------")


    rules = association_rules(frequent_itemsets, metric = 'confidence', 
                                min_threshold = 0.55)
    html_table_2 = rules.to_html(justify=CENTER,index=False,classes="table table-bordered dt-responsive",table_id="datatable_wrapper_2") 
    # Convert rules into coordinates suitable for use in a parallel coordinates plot
    coords = rules_to_coordinates(rules.head(40))

    fig=pltnew.figure(figsize=(10,5))
    parallel_coordinates(coords, 'rule')
    pltnew.legend([])
    pltnew.grid(True)
    pltnew.title(' Parallel Coordinates to Visualize Rules', fontsize=15,color="#0094cb",loc='left')
    flikes4 = BytesIO()
    pltnew.savefig(flikes4,bbox_inches='tight')
    pltnew.tight_layout()
    b645 = base64.b64encode(flikes4.getvalue()).decode()
    pltnew.close()
  
    st.write(" ")
    st.pyplot(fig)


    df=pd.read_csv("Market basket analysis data.csv")
    df
    df.Design= df.Design.str.lower()
    df["Design"]=df["Design"].astype('category')
    df["QUANTITY"]=1
    df2=df[["VOCNO","Design",'QUANTITY']]
    df2
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    
    basket=df2.groupby(["VOCNO","Design"])["QUANTITY"].sum().unstack().reset_index().fillna(0).set_index("VOCNO")
    basket=pd.DataFrame(basket)
    basket.head(300)
    
    basket_set = basket.applymap(encode_unit)
    basket_set
  
    color = pltnew.cm.rainbow(np.linspace(0, 1, 40))
    df['Design'].value_counts().head(40).plot.bar(color = color, figsize=(13,5))
    pltnew.title('Frequency of Most Popular Items', fontsize = 20)
    pltnew.xticks(rotation = 90 )
    pltnew.grid()
    #pltnew.show()
    flikes5 = BytesIO()
    pltnew.savefig(flikes5,bbox_inches='tight')
    pltnew.tight_layout()
    b646 = base64.b64encode(flikes5.getvalue()).decode()
    pltnew.close()

    import networkx as nx
    df['diamond'] = 'diamond ring'
    diamond = df.truncate(before = -1, after = 70)
    diamond = nx.from_pandas_edgelist(diamond, source = 'diamond', target = 'Design', edge_attr = True)

    import warnings
    warnings.filterwarnings('ignore')

    pltnew.rcParams['figure.figsize'] = (13, 11)
    pos = nx.spring_layout(diamond)
    color = pltnew.cm.Set1(np.linspace(0, 40, 1))
    nx.draw_networkx_nodes(diamond, pos, node_size = 12000, node_color = color)
    nx.draw_networkx_edges(diamond, pos, width = 2, alpha = 0.6, edge_color = 'black')
    nx.draw_networkx_labels(diamond, pos, font_size = 12, font_family = 'sans-serif')
    pltnew.axis('off')
    pltnew.grid()
    pltnew.title('Top 15 First Choices', fontsize = 20)
    #pltnew.show()
    flikes6 = BytesIO()
    pltnew.savefig(flikes6,bbox_inches='tight')
    pltnew.tight_layout()
    b647 = base64.b64encode(flikes6.getvalue()).decode()
    pltnew.close()

    frequent_itemsets = apriori(basket_set, min_support=0.08, use_colnames=True)
    #print (frequent_itemsets)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules.head()

    listToStr = ','.join([str(elem) for elem in dess])

    result = "SUCCESS"
    responses = {
            'wind_rose': b641,
            'chart2':b642,
            'chart3':b643,
            'chart4':b644,
            'chart5':b645,
            'chart6':b646,
            'chart7':b647,
            'tablesdata':html_table,
            'tablesdata2':html_table_2,
            'type':listToStr,
            "Status": result,
            "name": name,

    }
    #responses = {'wind_rose': "b641"}
    return JsonResponse(responses)       

def handle_uploaded_file(f):  
    with open('files/'+f.name, 'wb+') as destination:  
        for chunk in f.chunks():  
            destination.write(chunk)   
def testcsv(request):

   
  
   


    # # In[4]:

    # # In[12]:


    # res = requests.get("http://brilliantbidata.sunwebapps.com/api/MarketBasket?strFromDate=2021/01/01&strTodate=today")


    # # In[13]:


    # j = res.json()
    # df = pd.DataFrame(j)


    # # In[14]:


    # df=df[df["VOCTYPE"].isin(["PS1","PSR"])]
    # df


    # # In[15]:


    # df['NETVALUECC1'] = df['NETVALUECC1'].astype(float)
    # df['PRICE_RANGE'] = pd.cut(x=df["NETVALUECC1"], bins=[0,2000,10000,20000,30000,40000,50000,60000,75000,600000],labels=['0-2000', '2000-10000', '10000-20000',"20000-30000","30000-40000","40000-50000","50000-60000","60000-75000","75000 and above"])
    #     #pd.cut(df['some_col'], bins=[0,20,40,60], labels=['0-20', '20-40', '40-60']) 
    # df=df.dropna()


    # # In[16]:


    # new={'PN':"diamond pendant",
    # 'RN':"diamond ring",
    # 'BT':"diamond bracelet",
    # 'ER':"diamond earring",
    # 'BN':"diamond bangle",
    # 'NK':"diamond necklace",
    # 'BR':"gold bracelet",
    # 'GB':"gold bracelet",
    # 'GN':"diamond necklae",
    # 'REP':"jewellery repairing",
    # 'CUF':"diamond cufflink",
    # 'GBT':"gold bracelet with color stone",
    # 'DIA':"gold chain",
    # 'AN':"diamond anklet", 
    # 'GE':"gold earring",
    # 'SUT':"diamond suiti",
    # 'GPN':"gold pendant with colour stone",
    # 'GER':"gold earring",
    # 'RP':"platinum ring",
    # 'GNK':"gold necklace",
    # 'NP':"nose pin", 
    # 'GBNC':'gold bangle with colour stone',
    # 'GHN':"gold hand chain",
    # 'BRCH':"gold brooch",
    # 'GP':"gold pendant",
    # 'JEW':"gold chain",
    # 'GRN':"gold ring with color stone",
    # 'CRN':"diamond crown",
    # 'HC':"hand chain",
    # 'DJEW':"cufflink", 
    # 'BB':"diamond belly button"}


    # # In[24]:


    # df.DESIGN_DESCRIPTION= df.DESIGN_DESCRIPTION.str.lower()
    # df["Design"]=df["DESIGN_DESCRIPTION"].astype('category')
    # #df["PRICE_RANGE"]=df["PRICE_RANGE"].astype('category')
    # df=df.replace({'CATEGORY_CODE':new})
    # df['Design'] = np.where(df['Design']== "diamond",df["CATEGORY_CODE"] , df['Design'])
    # df["QUANTITY"]=1
    # #df['VOCNO'] = pd.Categorical(df['VOCNO']
    # #df['VOCNO'] = df['VOCNO'].cat.add_categories('Unknown')
    # df2=df[["VOCNO","Design","PRICE_RANGE",'QUANTITY',"GROSS_WT1"]]
    # df2['VOCNO'] = df2['VOCNO'].astype(float)
    # df2['GROSS_WT1'] = df2['GROSS_WT1'].astype(float)
    # df2['GROSS_WT1'] = df2['GROSS_WT1']

    # #df2=df[["VOCNO","Design",'QUANTITY',"GROSS_WT1"]]

    # #df2['VOCNO'] = pd.Categorical(df['VOCNO'],categories=df['VOCNO'].dropna().unique())
    # #df2.to_csv("file one.csv")
    # df2 = df2.astype({"PRICE_RANGE": object})


    # # In[25]:


    # basket=df2.groupby(["VOCNO","PRICE_RANGE"])["GROSS_WT1"].sum().unstack().reset_index().fillna(0).set_index("VOCNO")
    # #basket=df2.groupby(["VOCNO","Design"])["GROSS_WT1"].sum().unstack().reset_index().fillna(0).set_index("VOCNO")
    # basket=pd.DataFrame(basket)
    # basket=basket.head(200)
    #     #basket.to_csv('file7.csv')


    # # In[26]:



    # def encode_unit(x):
    #     if x <= 0:
    #         return 0
    #     if x >= 1:
    #         return 1
            


    # # In[27]:


    # basket_set = basket.applymap(encode_unit)
    #     #basket_set.to_csv('file4.csv')
    # basket_set.dropna()


    # # In[29]:


    # frequent_itemsets = apriori(basket_set, min_support=0.08, use_colnames=True)
    #     #print (frequent_itemsets)
    # rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    # rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
    # rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
    # rules
    # rules
    # print(rules)


    # # In[59]:


    # support=rules['support'].values
    # confidence=rules['confidence'].values
    # import random


    # # In[60]:


    # for i in range (len(support)):
    #     support[i] = support[i] + 0.0025 * (random.randint(1,10) - 5) 
    #     confidence[i] = confidence[i] + 0.0025 * (random.randint(1,10) - 5)
    
    # pltnew.scatter(support, confidence,   alpha=0.5, marker="o")
    # pltnew.xlabel('support')
    # pltnew.ylabel('confidence') 
    # pltnew.show()


    # # In[91]:


    # rules


    # # In[92]:


    # A="10000-20000"


    # # In[93]:


    # new=rules.loc[rules.antecedents == A]


    # # In[94]:


    # new


    # # In[95]:


    # new1=new.sort_values(by=['lift'], ascending=False)


    # # In[96]:


    # new1


    # # In[97]:


    # new1=new1.head(2)


    # # In[98]:


    # new_list=new1['consequents'].tolist()


    # # In[99]:


    # new_list


    # # In[100]:


    # B= new_list[0]
    # C = new_list[1]


    # # In[102]:


    # print("the recomended prodect for ", A ,"IS ",B," ",C)


    # # In[ ]:





    # # In[ ]:





    # # In[ ]:





    # # In[ ]:



    # rules.columns = map(str.upper, rules.columns)

    # rules.to_csv('rules.csv')


    # # In[44]:




    # rules["SUPPORT"] = pd.Series([round(val, 2) for val in rules["SUPPORT"]], index = rules.index)
    # rules["SUPPORT"]=rules["SUPPORT"]*100


    # # In[45]:


    # def rules_to_coordinates(rules):
    #     rules['antecedent'] = rules['antecedents'].apply(lambda antecedent: list(antecedent)[0])
    #     rules['consequent'] = rules['consequents'].apply(lambda consequent: list(consequent)[0])
    #     rules['rule'] = rules.index
    #     return rules[['antecedent','consequent','rule']]


    # # In[46]:


    # from pandas.plotting import parallel_coordinates

    # # Compute the frequent itemsets
    # #frequent_itemsets = apriori(onehot, min_support = 0.15, 
    # #                           use_colnames = True, max_len = 2)

    # # Compute rules from the frequent itemsets
    # rules = association_rules(frequent_itemsets, metric = 'confidence', 
    #                         min_threshold = 0.55)

    # # Convert rules into coordinates suitable for use in a parallel coordinates plot
    # coords = rules_to_coordinates(rules.head(40))

    # # Generate parallel coordinates plot
    # pltnew.figure(figsize=(4,8))
    # parallel_coordinates(coords, 'rule')
    # pltnew.legend([])
    # pltnew.grid(True)
    # pltnew.show()


    # # In[47]:


    # rules = association_rules(frequent_itemsets, metric='support', 
    #                         min_threshold = 0.0)

    # # Generate scatterplot using support and confidence
    # pltnew.figure(figsize=(10,6))
    # sns.scatterplot(x = "support", y = "confidence", data = rules)
    # pltnew.margins(0.01,0.01)
    # pltnew.show()


    # # In[48]:


    # pltnew.figure(figsize=(10,6))
    # sns.scatterplot(x = "support", y = "confidence", 
    #                 size = "lift", data = rules)
    # pltnew.margins(0.01,0.01)
    # pltnew.show()


    # # In[49]:


    # import plotly.express as px
    # from pandas.plotting import parallel_coordinates


    #     # Compute the frequent itemsets
    #     #frequent_itemsets = apriori(onehot, min_support = 0.15, 
    #     #                           use_colnames = True, max_len = 2)

    #     # Compute rules from the frequent itemsets
    # rules = association_rules(frequent_itemsets, metric = 'confidence', 
    #                         min_threshold = 0.55)

    #     # Convert rules into coordinates suitable for use in a parallel coordinates plot
    # coords = rules_to_coordinates(rules.head(40))

    #     # Generate parallel coordinates plot
        
    # fig=px.parallel_coordinates(rules, ['antecedents', 'consequents'] )
    # #parallel_coordinates(coords, 'rule')
    # #px.legend([])
    # #px.grid(True)
    # #px.title(' parallel coordinates to visualize rules', fontsize=15,color="blue")
    # fig.show()


    # # In[50]:


    # rules.columns =["antecedents","consequents",'antecedentSupport','consequentSupport',"support","confidence","lift","leverage","conviction"]


    # # In[51]:


    # rules


    # # In[53]:


    # pltnew.rcParams['figure.figsize'] = (10,6)
    # color = pltnew.cm.inferno(np.linspace(0,1,20))
    # rules['antecedents'].value_counts().head(20).plot.bar(color = color)
    # pltnew.title('Top 20 Most Frequent Items')
    # pltnew.ylabel('Counts')
    # pltnew.xlabel('Items')
    # pltnew.show()


    return render(request, template_name='pages/utility/chart3.html', context={'wind_rose': "",})