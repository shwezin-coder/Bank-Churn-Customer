import streamlit as st
import pandas as pd 
import numpy as np
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
st.set_page_config(layout="wide")

folder = 'dataset'
@st.cache_data
def load_data():
    return pd.read_csv(f'{folder}/cleaned_data.csv')
df = load_data()
    

def number_format(number):
    if number > 1000000:
        number = f"{round(number/1000000)} M"
    if number > 1000:
        number = f"{round(number/1000)} K"
    return number

def line_chart(x,y,chart_title=''):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x,y=y,text=y,mode="lines+markers+text",textposition='top right',hoverinfo='skip'))
    fig.update_layout(title=chart_title,xaxis_title='',yaxis_title='',yaxis= dict(showticklabels=False,showgrid=False))
    st.plotly_chart(fig,use_container_width=True)

def stack_bar_chart(data,color_map,chart_title='',showlegend=True):
    fig = go.Figure()
    for col in data.columns:
        fig.add_trace(go.Bar(x=data.index,y=data[col],text=data[col],hoverinfo='skip',name=col,marker_color=color_map[col]))
    fig.update_layout(title=chart_title,xaxis_title='',yaxis_title='',yaxis= dict(showticklabels=False,showgrid=False),barmode='stack',showlegend=showlegend)
    st.plotly_chart(fig,use_container_width=True)

def bar_chart(x,y,chart_title=''):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x,y=y,text=y,hoverinfo='skip'))
    fig.update_layout(title=chart_title,xaxis_title='',yaxis_title='',yaxis= dict(showticklabels=False,showgrid=False))
    st.plotly_chart(fig,use_container_width=True)

def horizontal_bar_chart(x,y,color_map,chart_title=''):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x,y=y,text=x,hoverinfo='skip',orientation='h',marker=dict(
        color=color_map
    )))
    fig.update_layout(title=chart_title,xaxis_title='',yaxis_title='',height=600,yaxis= dict(showgrid=False),xaxis=dict(showgrid=False,showticklabels=False))
    st.plotly_chart(fig,use_container_width=True)

def pie_chart(x,y,color_map,total_customers,chart_title=''):
    fig = go.Figure()
    fig.add_trace(go.Pie(labels=x,values=y,textinfo='percent+value',hole=0.5,marker_colors=color_map))
    fig.update_layout(title_text=chart_title,annotations=[dict(text=total_customers,x=0.5,y=0.5,showarrow=False)])
    st.plotly_chart(fig,use_container_width=True)

def render_filter(df,filter_columns,filter_layout = None):
    column_iter = iter(filter_layout) if filter_layout else None
    active_status_display = {'active':1,'inactive':0}
    churn_status_display = {'churner':1,'non churner':0}
    tenure_display = {x: f"{x} year" if x == 1 else f"{x} years" for x in sorted(df['tenure'].unique())}
    filter_values = {}
    for fc in filter_columns:
        with next(column_iter) if column_iter else st.container():
            if fc == 'tenure':
                filter_values['tenure'] = st.selectbox('Select Tenure',['all']+ list(tenure_display.keys()))
            elif fc == 'geography':
                filter_values['geography'] = st.selectbox('Select Geography',['all']+df['geography'].unique().tolist())
            elif fc == 'gender':
                filter_values['gender'] = st.selectbox('Select Gender',['all']+ df['gender'].unique().tolist())
            elif fc == 'isactivemember':
                filter_values['isactivemember'] = st.selectbox('Select Active Status',['all']+ list(active_status_display.keys()))
            elif fc == 'exited':
                filter_values['exited'] = st.selectbox('Select Churn Status',['all']+ list(churn_status_display.keys()))
            elif fc == 'numofproducts':
                filter_values['numofproducts'] = st.selectbox('Select Product Holdings',['all']+ df['numofproducts'].unique().tolist())
            elif fc == 'age_group':
                filter_values['age_group'] = st.selectbox('Select Age Group',['all']+ df['age_group'].unique().tolist())
            elif fc == 'creditscore_group':
                filter_values['creditscore_group'] = st.selectbox('Select Credit Score Group',['all']+ df['creditscore_group'].unique().tolist())
            elif fc == 'saving_group':
                filter_values['saving_group'] = st.selectbox('Select Saving Group',['all']+ df['saving_group'].unique().tolist())
            elif fc == 'estimatedsalary_group':
                filter_values['estimatedsalary_group'] = st.selectbox('Select Income Group',['all']+ df['estimatedsalary_group'].unique().tolist())
            else:
                st.warning(f'Filter {fc} is not defined')
        filtered_df = df.copy()

    for col,selected_value in filter_values.items():
        if col == 'tenure':
            selected_value = tenure_display.get(selected_value,selected_value)
        if col == 'isactivemember':
            selected_value = active_status_display.get(selected_value,selected_value)
        if col == 'exited':
            selected_value = churn_status_display.get(selected_value,selected_value)
        if selected_value != 'all':
            filtered_df = filtered_df[filtered_df[col]==selected_value]
    return filtered_df
header_col,nav_bar = st.columns([3,7])
with header_col:
    st.write('## Bank Churn Analysis')
with nav_bar:
    pages = ['Customer Overview','Churn Analysis','Churn Prediction','Strong Engagement Customers']
    nav_col = st.columns(len(pages))

    for col,pagename in zip(nav_col,pages):
        with col:
            if st.button(pagename):
                st.query_params.update(page=pagename)


query_params = st.query_params
page = query_params.get("page","Customer Overview")

if page == "Customer Overview":
    filter_pane,visual_pane,recommendation_pane = st.columns([1,7,2])
    with filter_pane:
        st.write("### Filter")
        filtered_df = render_filter(df,['tenure','geography','gender','isactivemember','exited','numofproducts','age_group','creditscore_group','saving_group','estimatedsalary_group'])
    with visual_pane:
        kpi1,kpi2,kpi3,kpi4,kpi5 = st.columns(5)
        kpi1.metric("Total Customers",len(filtered_df))
        kpi2.metric("Active Rate",f"{round(filtered_df['isactivemember'].mean()*100)}%")
        kpi3.metric("Avg Credit Score",f"{round(filtered_df['creditscore'].mean())}")
        kpi4.metric("Avg Saving (€)",f"{number_format(filtered_df[filtered_df['balance']>0]['balance'].mean())}")
        kpi5.metric("Avg Income (€)",f"{number_format(filtered_df['estimatedsalary'].median())}")

        tenure_customer_data = filtered_df['tenure'].value_counts().reset_index()
        tenure_customer_data.columns = ['Tenure','Customer Count']
        tenure_customer_data = tenure_customer_data.sort_values(by='Tenure')
        tenure_customer_data['Tenure'] = tenure_customer_data['Tenure'].apply(lambda x:f"{x} year" if x <= 1 else f"{x} years")
        
        credit_active_data = pd.crosstab(filtered_df['hascrcard'],filtered_df['isactivemember'],rownames=['Credit Card Holder'],colnames = ['Active Status']).reset_index()
        credit_active_data = credit_active_data.rename(index={0:'no credit card holder',1:'credit card holder'})
        credit_active_data = credit_active_data.rename(columns={0:'inactive',1:'active'})
        credit_active_data = credit_active_data.drop(columns='Credit Card Holder')

        age_group_data = filtered_df['age_group'].value_counts().reset_index()
        age_group_data.columns = ['Age Group','Customer Count']
        age_group_data = age_group_data.sort_values(by='Age Group')

        geo_saving_data = pd.crosstab(filtered_df['geography'],filtered_df['saving'],rownames=['Geography'],colnames=['Saving Status'])
        geo_saving_data = geo_saving_data.rename(columns={0:'no saving',1:'saving'})

        products_gender_data = pd.crosstab(filtered_df['numofproducts'],filtered_df['gender'],rownames=['Product Holder'],colnames=['Gender']).reset_index()
        products_gender_data.index = products_gender_data['Product Holder'].apply(lambda x:f"{x} Product Holder" if x == 1 else f"{x} Product Holders")
        products_gender_data = products_gender_data.drop(columns='Product Holder')

        creditscore_group_data = filtered_df['creditscore_group'].value_counts().reset_index()
        creditscore_group_data.columns = ['Credit Score Group','Customer Count']
        creditscore_group_data['Sorting'] = creditscore_group_data['Credit Score Group'].map({'below 650':1,'650-700':2,'above 700':3})
        creditscore_group_data = creditscore_group_data.sort_values(by='Sorting')
        creditscore_group_data = creditscore_group_data.drop(columns='Sorting')

        saving_group_data = filtered_df['saving_group'].value_counts().reset_index()
        saving_group_data.columns = ['Saving Group','Customer Count']
        saving_group_data['Sorting'] = saving_group_data['Saving Group'].map({'no saving':1, 'below 50k':2, '50k-1 lakhs' :3, '1 lakhs - 1.5 lakhs' : 4, '1.5 lakhs - 2 lakhs' :5, 'above 2 lakhs':6})
        saving_group_data = saving_group_data.sort_values(by='Sorting')
        saving_group_data = saving_group_data.drop(columns='Sorting')
        row1_left_chart_title,row1_right_chart_title, row1_detail_button = st.columns([2,1,1])

        income_group_data = filtered_df['estimatedsalary_group'].value_counts().reset_index()
        income_group_data.columns = ['Income Group','Customer Count']
        income_group_data['Sorting'] = income_group_data['Income Group'].map({'below 1k' :1, '1k-20k' :2, '20k-50k' :3, '50k - 1 lakh' :4, '1 lakh - 1.5 lakhs' :5, '1.5 lakhs - 2 lakhs':6, 'above 2 lakhs':7})
        income_group_data = income_group_data.sort_values(by='Sorting')
        income_group_data = income_group_data.drop(columns='Sorting')
        with row1_left_chart_title:
            st.write("Customers by Tenure")
        with row1_right_chart_title:
            st.write("Active vs Inactive Cardholders")
        with row1_detail_button:
            if st.button('Potential Active Customers'):
                st.query_params.update(page='Potential Active Customers')
        row1_left_chart,row1_right_chart = st.columns([1,1])
        with row1_left_chart:
            line_chart(tenure_customer_data['Tenure'],tenure_customer_data['Customer Count'])
        with row1_right_chart:
            color_map = {'inactive':'lightgrey','active':'skyblue'}
            stack_bar_chart(credit_active_data,color_map)
        row2_left_chart_title,row2_right_chart_title, row2_detail_button = st.columns([2,1,1])
        with row2_left_chart_title:
            st.write("Customers by Age Group")
        with row2_right_chart_title:
            st.write("Geography by Saving Status")
        with row2_detail_button:
            if st.button('Potential Saving Customers'):
                st.query_params.update(page='Potential Saving Customers')
        row2_left_chart,row2_right_chart = st.columns([1,1])
        with row2_left_chart:
            bar_chart(age_group_data['Age Group'],age_group_data['Customer Count'])
        with row2_right_chart:
            color_map = {'no saving':'lightgrey','saving':'skyblue'}
            stack_bar_chart(geo_saving_data,color_map)
        col_space,creditscore_group_btn,saving_group_btn, income_group_btn = st.columns([3,1,1,1])
        if 'selected_group' not in st.session_state:
            st.session_state.selected_group = "credit"
        with col_space:
            pass
        with creditscore_group_btn:
            if st.button('Credit Group',key='creditbtn'):
                st.session_state.selected_group = "credit"
        with saving_group_btn:
            if st.button('Saving Group',key='savingbtn'):
                st.session_state.selected_group = "saving"
        with income_group_btn:
            if st.button('Income Group',key='incomebtn'):
                st.session_state.selected_group = "income"
        row3_left_chart,row3_right_chart = st.columns([1,1])
        with row3_left_chart:
            color_map = {'female':'lightblue','male':'skyblue'}
            stack_bar_chart(products_gender_data,color_map,chart_title='Product Holders by Gender')
        with row3_right_chart:
            option = st.session_state.selected_group
            if option == 'credit':
                bar_chart(creditscore_group_data['Credit Score Group'],creditscore_group_data['Customer Count'],chart_title='Customers by Credit Score Group')
            if option == 'saving':
                bar_chart(saving_group_data['Saving Group'],saving_group_data['Customer Count'],chart_title='Customers by Saving Group')
            if option == 'income':
                bar_chart(income_group_data['Income Group'],income_group_data['Customer Count'],chart_title='Customers by Income Group')
    with recommendation_pane:
        st.markdown("""
### 🔍 Key Findings & Actions


#### 💳 Credit Card Activity
- **Active rate: 52%**
  - Low for a modern bank, acceptable for traditional.
  - Strong link between activity and card ownership.

**➡️ Action:** Boost engagement with a targeted **credit card offering**.


#### 👥 Customer Profile
- Avg. savings/income: **€100k**, credit score: **~650**
- **33–44 age group** = 50% of base  
- Full age range: **24–60**

**➡️ Action:** Focus **marketing/product** on age 33–44.


#### 🌍 Geography
- **Germany**: Strong savers  
- **Spain & France**: Untapped potential

**➡️ Action:** Run **savings campaigns** in Spain & France.

#### 📦 Product Holding
- Most hold **1–2 products**  
- **Females**: More likely to hold **3+ products**

**➡️ Action:**  
Cross-sell more to **females**.  
Encourage **males** to add 1 more product.
""")

if page == "Churn Analysis":
    geography_filter,age_filter,tenure_filter,numofproducts_filter,gender_filter = st.columns(5)
    filter_df = render_filter(df,['tenure','geography','gender','numofproducts','age_group'],filter_layout=[geography_filter,age_filter,tenure_filter,numofproducts_filter,gender_filter])
    strong_engagement_customers = filter_df[(filter_df['creditscore'] > 700) &
                                     (filter_df['balance'] > 100000) &
                                     (filter_df['estimatedsalary'] > 100000) &
                                     (filter_df['isactivemember'] == 1) 
                                    ]
    strong_engagement_churn_customers =  filter_df[(filter_df['creditscore'] > 700) &
                                     (filter_df['balance'] > 100000) &
                                     (filter_df['estimatedsalary'] > 100000) &
                                     (filter_df['isactivemember'] == 1) &
                                     (filter_df['exited'] == 1)
                                    ]
    churn_customers = filter_df[filter_df['exited'] == 1]

    strong_engagement_churn_rate_in_churn_customers = f'{round(len(strong_engagement_churn_customers)/len(churn_customers)*100,2)}%' # to fix division zero error 
    strong_engagement_churn_rate_in_engagment_customers = f'{round(len(strong_engagement_churn_customers)/len(strong_engagement_customers)*100,2)}%' # to fix division zero error
    metric1,metric2,very_strong_influence_btn,moderate_influence_btn,low_influence_btn,no_influence_btn = st.columns(6)
    metric1.metric('High Engagement (Churners) %',strong_engagement_churn_rate_in_churn_customers)
    metric2.metric('Strong Engaged Churn %',strong_engagement_churn_rate_in_engagment_customers)
    #len(strong_engagement_churn_customers),len(strong_engagement_customers)
    churn_categorical_columns = ['gender','geography','isactivemember','age_group','tenure','numofproducts','saving_group','creditscore_group','estimatedsalary_group']
    contingency_table = {}
    for col in churn_categorical_columns:
        contingency_table[col] = pd.crosstab(df[col],df['exited'])
    chi2_results = []
    for col in churn_categorical_columns:
        chi2_stats,chi2_p_val,dof,expected = chi2_contingency(contingency_table[col])
        if chi2_stats > 1000:
            influence_type = 'very strong'
        elif chi2_stats > 500:
            influence_type = 'strong'
        elif chi2_stats > 100:
            influence_type = 'moderate'
        elif chi2_p_val < 0.05:
            influence_type = 'low'
        else:
            influence_type = 'no'
        chi2_results.append({'variable':col,'p_val':chi2_p_val,'stats':chi2_stats,'influence_type':influence_type})
    chi2_results = pd.DataFrame(chi2_results).sort_values(by='p_val')
    chi2_results['p_val'] = chi2_results['p_val'].apply(lambda x:f'{x:.2e}')
    if 'influence_option' not in st.session_state:
        st.session_state.influence_option = "very strong"
    with very_strong_influence_btn:
        if st.button('Very Strong Influence',key='very_strong_btn'):
            st.session_state.influence_option = "very strong"
   # with strong_influence_btn:
   #     if st.button('Strong Influence',key='strong_btn'):
    #        st.session_state.influence_option = "strong"
    with moderate_influence_btn:
        if st.button('Moderate Influence',key='moderate_btn'):
            st.session_state.influence_option = "moderate"
    with low_influence_btn:
        if st.button('Low Influence',key='low_btn'):
            st.session_state.influence_option = "low"
    with no_influence_btn:
        if st.button('No Influence',key='no_btn'):
            st.session_state.influence_option = "no"
    left_section,right_section = st.columns([2,5])
    influence_option = st.session_state.influence_option 
    chi2_filter = chi2_results[chi2_results['influence_type'] == influence_option]
    with left_section:
        churn_status_data = filter_df['exited'].value_counts().reset_index()
        churn_status_data.columns = ['Churn Status','Customer Count']
        churn_status_data = churn_status_data.rename(index={0:'Non Churner',1:'Churner'})
        color_map = ['skyblue','lightgrey']
        pie_chart(churn_status_data.index,churn_status_data['Customer Count'],color_map,f'Total Customers<br>{len(df)}',chart_title='Churners vs Non Churners')
        engagement_geography_age_data = pd.crosstab(strong_engagement_churn_customers['age_group'],strong_engagement_churn_customers['geography'],rownames=['Age Group'],colnames=['Geographhy'])
        color_map = {'france':'royalblue','germany':'skyblue','spain':'lightblue'}
        stack_bar_chart(engagement_geography_age_data,color_map,chart_title='Engaged Churn by Age & Region')
    with right_section:
        st.subheader(f'Churn Distribution by Significance Level ({influence_option.capitalize()})')
        col_count = st.columns(len(chi2_filter))
        for col,variable in zip(col_count,chi2_filter['variable']):
            with col:
                influence_chart_data = pd.crosstab(df[variable],df['exited'],rownames=[variable],colnames=['Churn Status']).reset_index()
                influence_chart_data = influence_chart_data.set_index(variable)
                if variable == 'isactivemember':
                    variable = 'Active Status'
                    influence_chart_data = influence_chart_data.rename(index={0:'inactive',1:'active'})
                influence_chart_data = influence_chart_data.rename(columns={0:'No Churner',1:'Churner'})
                color_map = {'No Churner':'skyblue','Churner':'lightgrey'}
                stack_bar_chart(influence_chart_data,color_map,chart_title= f'{variable.replace('_',' ').capitalize()}',showlegend=False)
        left_col,right_col = st.columns([2,1])
        with left_col:
            st.subheader('Hypothesis Test Results')
            st.write(chi2_filter.drop(columns=['influence_type']))
            high_values_lost_customers = strong_engagement_churn_customers[['customerid','surname', 'geography', 'gender', 'age', 'numofproducts',  'tenure','balance', 
            'creditscore','estimatedsalary']].reset_index(drop=True)
            st.subheader(f"High Value Lost Customers - {len(high_values_lost_customers)}")
            st.write(high_values_lost_customers)
        with right_col:
            st.markdown("""
### 🔍 Churn Insights & Actions

**Key Factors (Chi² Test):**  
- **Strong (Chi² > 1000):** Age Group, Product Count  
- **Moderate (100–300):** Geography, Active Saving, Gender  
- **Low:** Credit Score  
- **Not Relevant:** Income, Tenure  

**Churn Metrics:**  
- **High-Value Churned:** 3.14%  
- **Engaged Churn Rate:** 15.88%  
- **Total Churn:** 20.4% *(within industry range)*  
- **Germany:** Higher than average — needs investigation  

**Actions:**  
- Focus churn model on **age** & **products**  
- Analyze **Germany**: competitor & service review  
- Use **High-Value Table** for detail  
- Assign to **Ops & Service** teams  
- ⚠️ Ensure **PII compliance**
""")



if page == "Churn Prediction":
    filter_pane,col_space,visual_pane,findings_pane = st.columns([2,1,4,2])
    with filter_pane:
        filtered_df = render_filter(df,['tenure','geography','gender','isactivemember','exited','numofproducts','age_group','creditscore_group','saving_group','estimatedsalary_group'])
    with visual_pane:
        st.markdown("<div style='margin-top:100px'></div>",unsafe_allow_html=True)
        kpi1,kpi2 = st.columns(2)
        kpi1.metric('Avg Churn Probability',f"{round(filtered_df['exit_proba'].mean()*100,2)}%")
        kpi2.metric('Avg Stay Probability',f"{round(filtered_df['stay_proba'].mean()*100,2)}%")

        feature_importance_data = pd.read_csv(f'{folder}/feature_coefficient.csv')
        feature_importance_data['coefficient'] = feature_importance_data['coefficient'].apply(lambda x:round(x,2))
        feature_importance_data = feature_importance_data.sort_values(by='coefficient',ascending=True)
        color_map = ['skyblue' if val < 0 else 'red' for val in feature_importance_data['coefficient']]
        horizontal_bar_chart(feature_importance_data['coefficient'],feature_importance_data['feature'],color_map,chart_title='Feature Influence on Churn Status')
    with findings_pane:
       st.markdown("""
### 🤖 Churn Model & Insights

**Model:** Logistic Regression  
- Chosen for feature clarity  
- **Accuracy:** 0.83, **F1:** 0.60  
- Predicted **62%** of churners — **missed ~38%**, so we need to improve the model to capture the rest.

**➡️ Action:**  
Focus on customers with **>70% churn risk** — all teams act early.  
Provide **credit card plans** to customers without cards — they are often inactive and more likely to churn. This can **boost engagement** and reduce churn.

**Key Churn Signals:**  
- **Products:** Females with more products = higher churn; 2-product holders = stable  
- **Geography:** Germany & Spain churn > France  
- **Age:** 33–44 & 45–60 more likely to churn  
- **Savings:** >200k or <50k = more churn; 50k–150k = more stable  
- **Credit Card:** Customers without a credit card are more likely to churn  

**Next Steps:**  
- Use shared table to act on segments  
- ⚠️ Always protect PII  
""")

    potential_high_risk_customers = filtered_df[(filtered_df['exit_proba']>=0.7) & (filtered_df['exited'] == 0)]
    st.subheader(f'Potential High Risk Customers - {len(potential_high_risk_customers)}')
    potential_high_risk_customers['active status'] = potential_high_risk_customers['isactivemember'].map({0:'inactive',1:'active'})
    potential_high_risk_customers = potential_high_risk_customers[['customerid','surname','active status','geography',
            'gender', 'age',   'numofproducts', 'tenure','balance',
            'creditscore', 'estimatedsalary']]
    st.write(potential_high_risk_customers)

if page == "Strong Engagement Customers":
    strong_engagement_customers = df[
                 (df['isactivemember'] == 1) &
                 (df['exited'] == 0) &
                 (df['creditscore'] >= 700) &
                 (df['estimatedsalary'] >= 100000) &
                 (df['balance'] > 100000)
                ]
    st.subheader('Strong Engagement Customers Analysis')
    left_section,right_section = st.columns([5,1])
    with left_section:
        row1_filter,col_space,row1_kpi_card,row1_chart = st.columns([1,1,2,4])
        with row1_filter:
            st.markdown("<div style='margin-top:30px'></div>",unsafe_allow_html=True)
            filtered_potential_df = render_filter(strong_engagement_customers,['geography','gender','numofproducts','age_group'])
        with row1_kpi_card:
            st.markdown("<div style='margin-top:100px'></div>",unsafe_allow_html=True)
            total_strong_engagement_customers = len(filtered_potential_df)
            strong_engagement_rate = f"{round(total_strong_engagement_customers/len(df)*100,2)}%"
            row1_kpi_card.metric('Total Strong Engagement Customers',total_strong_engagement_customers)
            row1_kpi_card.metric('Strong Engagement Rate',strong_engagement_rate)
        with row1_chart:
            potential_age_group_gender = pd.crosstab(filtered_potential_df['age_group'],filtered_potential_df['gender'],rownames=['Age Group'],colnames=['Gender']).reset_index()
            potential_age_group_gender.index = potential_age_group_gender['Age Group']
            potential_age_group_gender = potential_age_group_gender.drop(columns='Age Group')
            color_map = {'female':'lightblue','male':'skyblue'}
            stack_bar_chart(potential_age_group_gender,color_map,chart_title='Strong Engagement Customers by Age Group and Gender')
        potential_table = filtered_potential_df[['customerid','surname','geography',
       'gender', 'age',   'numofproducts', 'tenure','balance',
        'creditscore', 'estimatedsalary']]
        st.write("Strong Engagement Customers List")
        st.write(potential_table)

    with right_section:
        st.markdown("""
### 🔍 High-Engagement Segment (3.4%)


#### 🧩 Profile
- **Active**, not churned  
- **Credit score > 700**  
- **Salary & balance > €100k**  
- **Age 24–60**, mostly **male**


#### 🎯 Action

- Ideal for **new product launches**, **campaigns**, or **promotions**
- Can be targeted by **Product**, **Marketing**, or **Sales** teams
- Detailed info available in the provided **customer table**

**⚠️ Reminder:**  
Use data responsibly — ensure **PII compliance** at all times.

""")
if page == "Potential Active Customers":
    potential_active_customers = df[(df['hascrcard']==0) &
                 (df['exited'] == 0) &
                 (df['creditscore'] >= 700) &
                 (df['estimatedsalary'] >= 100000) &
                 (df['balance'] >= 100000)
                ]
    st.subheader("Potential Active Customers Analysis")
    left_section,right_section = st.columns([5,1])
    with left_section:
        row1_filter,col_space,row1_kpi_card,row1_chart = st.columns([1,1,2,4])
        with row1_filter:
            st.markdown("<div style='margin-top:30px'></div>",unsafe_allow_html=True)
            filtered_potential_df = render_filter(potential_active_customers,['geography','gender','numofproducts','age_group'])
        with row1_kpi_card:
            st.markdown("<div style='margin-top:100px'></div>",unsafe_allow_html=True)
            total_potential_active_customers = len(filtered_potential_df)
            potential_active_rate = f"{round(total_potential_active_customers/len(df)*100,2)}%"
            row1_kpi_card.metric('Total Potential Active Customers',total_potential_active_customers)
            row1_kpi_card.metric('Potential Active Rate',f'+{potential_active_rate}')
        with row1_chart:
            potential_geography_gender = pd.crosstab(filtered_potential_df['geography'],filtered_potential_df['gender'],rownames=['Geography'],colnames=['Gender']).reset_index()
            potential_geography_gender.index = potential_geography_gender['Geography']
            potential_geography_gender = potential_geography_gender.drop(columns='Geography')
            color_map = {'female':'lightblue','male':'skyblue'}
            stack_bar_chart(potential_geography_gender,color_map,chart_title='Potential Customers by Geography and Gender')
        potential_table = filtered_potential_df[['customerid','surname','geography',
       'gender', 'age',   'numofproducts', 'tenure','balance',
        'creditscore', 'estimatedsalary']]
        potential_table['balance'] = potential_table['balance'].apply(lambda x:f"€ {round(x)}")
        potential_table['estimatedsalary'] = potential_table['estimatedsalary'].apply(lambda x:f"€ {round(x)}")
        st.write("Potential Customers List")
        
        st.write(potential_table)
    with right_section:
        st.markdown("""
### 🔍 Key Findings & Actions

#### 🎯 High-Potential Segment (2.43%)
- **Credit > 700**, ~€100k salary & savings  
- **Inactive but retained**

**➡️ Action:**  
Target with **credit card offers** — 100% link to activity.

#### 👤 Gender & Region
- **France/Spain**: More **males**  
- **Germany**: More **females**  
- Key age group: **33–44**

**➡️ Action:**  
Tailor campaigns by **gender + region**, focus on **33–44**.

#### 🧭 Next Steps
- Use **Potential Customers Table**  
- Teams: Marketing / Sales / Key Account

**⚠️ Respect PII at all stages**
""")
      

if page == "Potential Saving Customers":
    potential_saving_customers = df[
                 (df['exited'] == 0) &
                 (df['creditscore'] >= 700) &
                 (df['estimatedsalary'] >= 100000) &
                 (df['balance'] == 0)
                ]
    st.subheader('Potential Saving Customers Analysis')
    left_section,right_section = st.columns([5,1])
    with left_section:
        row1_filter,col_space,row1_kpi_card,row1_chart = st.columns([1,1,2,4])
        with row1_filter:
            st.markdown("<div style='margin-top:30px'></div>",unsafe_allow_html=True)
            filtered_potential_df = render_filter(potential_saving_customers,['geography','gender','isactivemember','numofproducts','age_group'])
        with row1_kpi_card:
            st.markdown("<div style='margin-top:100px'></div>",unsafe_allow_html=True)
            total_potential_saving_customers = len(filtered_potential_df)
            potential_saving_rate = f"{round(total_potential_saving_customers/len(df)*100,2)}%"
            row1_kpi_card.metric('Total Potential Saving Customers',total_potential_saving_customers)
            row1_kpi_card.metric('Potential Saving Rate',potential_saving_rate)
        with row1_chart:
            potential_age_group_gender = pd.crosstab(filtered_potential_df['age_group'],filtered_potential_df['gender'],rownames=['Age Group'],colnames=['Gender']).reset_index()
            potential_age_group_gender.index = potential_age_group_gender['Age Group']
            potential_age_group_gender = potential_age_group_gender.drop(columns='Age Group')
            color_map = {'female':'lightblue','male':'skyblue'}
            stack_bar_chart(potential_age_group_gender,color_map,chart_title='Potential Customers by Age Group and Gender')
        filtered_potential_df['active status'] = filtered_potential_df['isactivemember'].map({0:'inactive',1:'active'})
        potential_table = filtered_potential_df[['customerid','surname','active status','geography',
       'gender', 'age',   'numofproducts', 'tenure','balance',
        'creditscore', 'estimatedsalary']]
        potential_table.columns = potential_table.columns.str.capitalize()
        st.write("Potential Customers List")
        st.write(potential_table)

    with right_section:
        st.markdown("""
### 🔍 Key Findings & Actions


#### 💡 High-Value Segment (4.8%)

- **Credit score > 700**, **Salary > €100k**
- Includes **active & inactive** customers

**➡️ Inactive:**  
Target with **credit card plans** to boost engagement

**➡️ Active:**  
Offer **strict loans** or **adjusted savings rates** to grow retention


#### 👥 Demographics

- Age: **24–60**, focus on **33–44**
- Matches main customer profile


#### 📢 Marketing Use

- Ideal for **personalized campaigns**
- Use the **Potential Customers Table** for targeting  
- **Respect PII** and privacy at all stages

""")
