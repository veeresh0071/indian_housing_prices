"""
Streamlit Application for Real Estate Investment Advisor
Provides property investment predictions and analysis dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# Page configuration
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models and data
@st.cache_resource
def load_models():
    """Load trained models"""
    clf_model = joblib.load('models/classification_model.pkl')
    reg_model = joblib.load('models/regression_model.pkl')
    model_info = joblib.load('models/model_info.pkl')
    return clf_model, reg_model, model_info

@st.cache_data
def load_data():
    """Load engineered dataset for reference values"""
    df = pd.read_csv('data/engineered_data.csv')
    return df

# Initialize
try:
    clf_model, reg_model, model_info = load_models()
    df = load_data()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Error loading models: {e}")

# Get unique values for dropdowns
if models_loaded:
    states = sorted(df['State'].unique())
    cities = sorted(df['City'].unique())
    property_types = sorted(df['Property_Type'].unique())
    furnished_status = sorted(df['Furnished_Status'].unique())
    facing_options = sorted(df['Facing'].unique())
    transport_options = sorted(df['Public_Transport_Accessibility'].unique())
    owner_types = sorted(df['Owner_Type'].unique())
    availability_options = sorted(df['Availability_Status'].unique())

# Helper functions
def get_cities_by_state(state):
    """Get cities for selected state"""
    return sorted(df[df['State'] == state]['City'].unique())

def calculate_infrastructure_score(parking, security, transport, schools, hospitals):
    """Calculate infrastructure score"""
    parking_val = 1 if parking == 'Yes' else 0
    security_val = 1 if security == 'Yes' else 0
    transport_map = {'High': 3, 'Medium': 2, 'Low': 1}
    transport_val = transport_map.get(transport, 2)
    
    score = (
        parking_val * 0.15 +
        security_val * 0.20 +
        (transport_val / 3) * 0.25 +
        (schools / 10) * 0.20 +
        (hospitals / 10) * 0.20
    )
    return score

def prepare_input_data(inputs):
    """Prepare input data for model prediction"""
    # Get encodings from data
    state_encoding = df[df['State'] == inputs['state']]['State_Encoded'].iloc[0]
    city_encoding = df[df['City'] == inputs['city']]['City_Encoded'].iloc[0]
    property_type_encoding = df[df['Property_Type'] == inputs['property_type']]['Property_Type_Encoded'].iloc[0]
    furnished_encoding = df[df['Furnished_Status'] == inputs['furnished']]['Furnished_Status_Encoded'].iloc[0]
    facing_encoding = df[df['Facing'] == inputs['facing']]['Facing_Encoded'].iloc[0]
    transport_encoding = df[df['Public_Transport_Accessibility'] == inputs['transport']]['Public_Transport_Accessibility_Encoded'].iloc[0]
    parking_encoding = df[df['Parking_Space'] == inputs['parking']]['Parking_Space_Encoded'].iloc[0]
    security_encoding = df[df['Security'] == inputs['security']]['Security_Encoded'].iloc[0]
    owner_encoding = df[df['Owner_Type'] == inputs['owner_type']]['Owner_Type_Encoded'].iloc[0]
    availability_encoding = df[df['Availability_Status'] == inputs['availability']]['Availability_Status_Encoded'].iloc[0]
    
    # Get city/state median prices
    city_median = df[df['City'] == inputs['city']]['City_Median_Price'].iloc[0]
    state_median = df[df['State'] == inputs['state']]['State_Median_Price'].iloc[0]
    
    # Calculate derived features
    price_per_sqft = inputs['price'] / inputs['size'] if inputs['size'] > 0 else 0
    infra_score = calculate_infrastructure_score(
        inputs['parking'], inputs['security'], inputs['transport'],
        inputs['schools'], inputs['hospitals']
    )
    amenities_count = sum([inputs['pool'], inputs['gym'], inputs['garden'], 
                          inputs['clubhouse'], inputs['playground']])
    
    return {
        'Property_Type_Encoded': property_type_encoding,
        'Furnished_Status_Encoded': furnished_encoding,
        'Facing_Encoded': facing_encoding,
        'Public_Transport_Accessibility_Encoded': transport_encoding,
        'Parking_Space_Encoded': parking_encoding,
        'Security_Encoded': security_encoding,
        'Owner_Type_Encoded': owner_encoding,
        'Availability_Status_Encoded': availability_encoding,
        'City_Encoded': city_encoding,
        'State_Encoded': state_encoding,
        'BHK': inputs['bhk'],
        'Size_in_SqFt': inputs['size'],
        'Age_of_Property': inputs['age'],
        'Floor_No': inputs['floor'],
        'Total_Floors': inputs['total_floors'],
        'Nearby_Schools': inputs['schools'],
        'Nearby_Hospitals': inputs['hospitals'],
        'Has_Pool': inputs['pool'],
        'Has_Gym': inputs['gym'],
        'Has_Garden': inputs['garden'],
        'Has_Clubhouse': inputs['clubhouse'],
        'Has_Playground': inputs['playground'],
        'Amenities_Count': amenities_count,
        'Infrastructure_Score': infra_score,
        'Price_in_Lakhs': inputs['price'],
        'Price_per_SqFt': price_per_sqft,
        'City_Median_Price': city_median,
        'State_Median_Price': state_median
    }


# Main App
def main():
    # Header
    st.title("🏠 Real Estate Investment Advisor")
    st.markdown("*AI-powered property investment analysis and price prediction*")
    
    if not models_loaded:
        st.error("Models not loaded. Please ensure model files exist in the 'models/' directory.")
        return
    
    # Sidebar - Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["🔮 Investment Predictor", "📊 Market Dashboard", "🔍 Property Filter"])
    
    if page == "🔮 Investment Predictor":
        show_predictor_page()
    elif page == "📊 Market Dashboard":
        show_dashboard_page()
    else:
        show_filter_page()


def show_predictor_page():
    """Investment prediction page with user input form"""
    st.header("Property Investment Analysis")
    st.markdown("Enter property details to get investment recommendation and 5-year price prediction.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📍 Location")
        state = st.selectbox("State", states, index=0)
        available_cities = get_cities_by_state(state)
        city = st.selectbox("City", available_cities, index=0)
        
        st.subheader("🏗️ Property Details")
        property_type = st.selectbox("Property Type", property_types)
        bhk = st.slider("BHK (Bedrooms)", 1, 5, 3)
        size = st.number_input("Size (Sq Ft)", min_value=500, max_value=5000, value=2000, step=100)
        age = st.slider("Property Age (Years)", 0, 35, 10)
    
    with col2:
        st.subheader("🏢 Building Info")
        floor = st.slider("Floor Number", 0, 30, 5)
        total_floors = st.slider("Total Floors", 1, 30, 15)
        facing = st.selectbox("Facing Direction", facing_options)
        furnished = st.selectbox("Furnished Status", furnished_status)
        availability = st.selectbox("Availability", availability_options)
        
        st.subheader("💰 Pricing")
        price = st.number_input("Price (in Lakhs ₹)", min_value=10.0, max_value=500.0, value=200.0, step=10.0)
    
    with col3:
        st.subheader("🏪 Amenities")
        pool = st.checkbox("Swimming Pool", value=False)
        gym = st.checkbox("Gym", value=False)
        garden = st.checkbox("Garden", value=False)
        clubhouse = st.checkbox("Clubhouse", value=False)
        playground = st.checkbox("Playground", value=False)
        
        st.subheader("🚗 Infrastructure")
        parking = st.selectbox("Parking Space", ['Yes', 'No'])
        security = st.selectbox("Security", ['Yes', 'No'])
        transport = st.selectbox("Public Transport", transport_options)
        schools = st.slider("Nearby Schools", 1, 10, 5)
        hospitals = st.slider("Nearby Hospitals", 1, 10, 5)
        owner_type = st.selectbox("Owner Type", owner_types)
    
    # Predict button
    if st.button("🔮 Analyze Investment", type="primary", use_container_width=True):
        # Prepare inputs
        inputs = {
            'state': state, 'city': city, 'property_type': property_type,
            'bhk': bhk, 'size': size, 'age': age, 'floor': floor,
            'total_floors': total_floors, 'facing': facing, 'furnished': furnished,
            'availability': availability, 'price': price, 'parking': parking,
            'security': security, 'transport': transport, 'schools': schools,
            'hospitals': hospitals, 'owner_type': owner_type,
            'pool': int(pool), 'gym': int(gym), 'garden': int(garden),
            'clubhouse': int(clubhouse), 'playground': int(playground)
        }
        
        # Prepare data
        prepared_data = prepare_input_data(inputs)
        
        # Classification prediction
        clf_features = model_info['clf_features']
        clf_input = pd.DataFrame([[prepared_data[f] for f in clf_features]], columns=clf_features)
        investment_pred = clf_model.predict(clf_input)[0]
        investment_prob = clf_model.predict_proba(clf_input)[0][1]
        
        # Regression prediction
        reg_features = model_info['reg_features']
        reg_input = pd.DataFrame([[prepared_data[f] for f in reg_features]], columns=reg_features)
        price_5y = reg_model.predict(reg_input)[0]
        
        # Display results
        st.markdown("---")
        st.header("📈 Analysis Results")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            if investment_pred == 1:
                st.success("✅ GOOD INVESTMENT")
                st.metric("Investment Score", f"{investment_prob*100:.1f}%", "Recommended")
            else:
                st.error("❌ NOT RECOMMENDED")
                st.metric("Investment Score", f"{investment_prob*100:.1f}%", "Risky")
        
        with res_col2:
            price_gain = price_5y - price
            gain_pct = (price_gain / price) * 100
            st.metric("Current Price", f"₹{price:.2f}L")
            st.metric("5-Year Predicted Price", f"₹{price_5y:.2f}L", f"+₹{price_gain:.2f}L ({gain_pct:.1f}%)")
        
        with res_col3:
            st.metric("Price per Sq Ft", f"₹{(price*100000/size):.0f}")
            city_median = prepared_data['City_Median_Price']
            price_vs_median = ((price - city_median) / city_median) * 100
            st.metric("vs City Median", f"{price_vs_median:+.1f}%", 
                     "Below Average" if price_vs_median < 0 else "Above Average")
        
        # Visualization
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.subheader("📊 Price Projection")
            years = list(range(6))
            annual_rate = (price_5y / price) ** (1/5) - 1
            projected_prices = [price * (1 + annual_rate) ** y for y in years]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=years, y=projected_prices, mode='lines+markers',
                                     name='Projected Price', line=dict(color='#2ecc71', width=3)))
            fig.update_layout(
                title='5-Year Price Projection',
                xaxis_title='Years from Now',
                yaxis_title='Price (₹ Lakhs)',
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_col2:
            st.subheader("🎯 Feature Importance")
            # Get feature importance from classification model
            if hasattr(clf_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': clf_features,
                    'Importance': clf_model.feature_importances_
                }).sort_values('Importance', ascending=True).tail(10)
                
                fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                                title='Top 10 Features for Investment Decision',
                                color='Importance', color_continuous_scale='Greens')
                fig_imp.update_layout(template='plotly_white', showlegend=False)
                st.plotly_chart(fig_imp, use_container_width=True)


def show_dashboard_page():
    """Market analysis dashboard"""
    st.header("📊 Market Dashboard")
    st.markdown("Explore real estate market trends and insights.")
    
    # Add tabs for different views
    tab1, tab2 = st.tabs(["📈 Market Analysis", "🗺️ Location Heatmap"])
    
    with tab1:
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_state = st.selectbox("Filter by State", ['All'] + states, key='dash_state')
        with col2:
            if selected_state != 'All':
                available_cities = ['All'] + get_cities_by_state(selected_state)
            else:
                available_cities = ['All'] + cities
            selected_city = st.selectbox("Filter by City", available_cities, key='dash_city')
        with col3:
            selected_type = st.selectbox("Property Type", ['All'] + property_types, key='dash_type')
        
        # Filter data
        filtered_df = df.copy()
        if selected_state != 'All':
            filtered_df = filtered_df[filtered_df['State'] == selected_state]
        if selected_city != 'All':
            filtered_df = filtered_df[filtered_df['City'] == selected_city]
        if selected_type != 'All':
            filtered_df = filtered_df[filtered_df['Property_Type'] == selected_type]
        
        # Key metrics
        st.subheader("📈 Key Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Properties", f"{len(filtered_df):,}")
        m2.metric("Avg Price", f"₹{filtered_df['Price_in_Lakhs'].mean():.2f}L")
        m3.metric("Avg Size", f"{filtered_df['Size_in_SqFt'].mean():.0f} sqft")
        m4.metric("Good Investments", f"{filtered_df['Good_Investment'].mean()*100:.1f}%")
        
        # Charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("Price Distribution")
            fig1 = px.histogram(filtered_df, x='Price_in_Lakhs', nbins=50,
                               title='Property Price Distribution',
                               labels={'Price_in_Lakhs': 'Price (₹ Lakhs)'})
            fig1.update_layout(template='plotly_white')
            st.plotly_chart(fig1, use_container_width=True)
            
            st.subheader("Price by Property Type")
            type_price = filtered_df.groupby('Property_Type')['Price_in_Lakhs'].mean().reset_index()
            fig3 = px.bar(type_price, x='Property_Type', y='Price_in_Lakhs',
                         title='Average Price by Property Type',
                         labels={'Price_in_Lakhs': 'Avg Price (₹ Lakhs)'})
            fig3.update_layout(template='plotly_white')
            st.plotly_chart(fig3, use_container_width=True)
        
        with chart_col2:
            st.subheader("Investment Grade Distribution")
            grade_counts = filtered_df['Investment_Grade'].value_counts().reset_index()
            grade_counts.columns = ['Grade', 'Count']
            fig2 = px.pie(grade_counts, values='Count', names='Grade',
                         title='Investment Grade Distribution',
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig2, use_container_width=True)
            
            st.subheader("Price vs Size")
            sample = filtered_df.sample(min(1000, len(filtered_df)), random_state=42)
            fig4 = px.scatter(sample, x='Size_in_SqFt', y='Price_in_Lakhs',
                             color='Property_Type', opacity=0.6,
                             title='Price vs Size by Property Type',
                             labels={'Size_in_SqFt': 'Size (Sq Ft)', 'Price_in_Lakhs': 'Price (₹ Lakhs)'})
            fig4.update_layout(template='plotly_white')
            st.plotly_chart(fig4, use_container_width=True)
    
        # Top cities
        st.subheader("🏙️ Top 10 Cities by Average Price")
        top_cities = filtered_df.groupby('City').agg({
            'Price_in_Lakhs': 'mean',
            'Good_Investment': 'mean',
            'ID': 'count'
        }).reset_index()
        top_cities.columns = ['City', 'Avg Price (₹L)', 'Good Investment %', 'Properties']
        top_cities['Good Investment %'] = (top_cities['Good Investment %'] * 100).round(1)
        top_cities = top_cities.sort_values('Avg Price (₹L)', ascending=False).head(10)
        st.dataframe(top_cities, use_container_width=True, hide_index=True)
    
    with tab2:
        # Location Heatmap
        st.subheader("🗺️ Price Heatmap by Location")
        
        # State-level heatmap
        state_stats = df.groupby('State').agg({
            'Price_in_Lakhs': 'mean',
            'Good_Investment': 'mean',
            'Predicted_Price_5Y': 'mean',
            'ID': 'count'
        }).reset_index()
        state_stats.columns = ['State', 'Avg Price', 'Good Investment %', 'Avg 5Y Price', 'Count']
        state_stats['Good Investment %'] = (state_stats['Good Investment %'] * 100).round(1)
        
        fig_heatmap = px.treemap(state_stats, path=['State'], values='Count',
                                  color='Avg Price', color_continuous_scale='RdYlGn_r',
                                  title='State-wise Property Distribution (Color = Avg Price)',
                                  hover_data=['Good Investment %', 'Avg 5Y Price'])
        fig_heatmap.update_layout(template='plotly_white')
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # City-level analysis
        st.subheader("🏙️ City-wise Price Analysis")
        city_stats = df.groupby(['State', 'City']).agg({
            'Price_in_Lakhs': 'mean',
            'Good_Investment': 'mean',
            'Price_Gain_Pct_5Y': 'mean',
            'ID': 'count'
        }).reset_index()
        city_stats.columns = ['State', 'City', 'Avg Price (₹L)', 'Good Investment %', '5Y Gain %', 'Properties']
        city_stats['Good Investment %'] = (city_stats['Good Investment %'] * 100).round(1)
        city_stats['5Y Gain %'] = city_stats['5Y Gain %'].round(1)
        city_stats['Avg Price (₹L)'] = city_stats['Avg Price (₹L)'].round(2)
        
        # Heatmap matrix
        heatmap_data = df.pivot_table(values='Price_in_Lakhs', index='State', 
                                       columns='Property_Type', aggfunc='mean')
        fig_matrix = px.imshow(heatmap_data, text_auto='.1f',
                               title='Average Price by State and Property Type (₹ Lakhs)',
                               color_continuous_scale='Blues',
                               labels=dict(color="Avg Price"))
        fig_matrix.update_layout(template='plotly_white')
        st.plotly_chart(fig_matrix, use_container_width=True)
        
        # City table
        st.dataframe(city_stats.sort_values('Avg Price (₹L)', ascending=False).head(20), 
                    use_container_width=True, hide_index=True)


def show_filter_page():
    """Property filtering and search page"""
    st.header("🔍 Property Filter")
    st.markdown("Find properties matching your criteria.")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        filter_state = st.multiselect("State", states, default=[])
        filter_type = st.multiselect("Property Type", property_types, default=[])
    
    with col2:
        price_range = st.slider("Price Range (₹ Lakhs)", 10, 500, (50, 300))
        size_range = st.slider("Size Range (Sq Ft)", 500, 5000, (1000, 3000))
    
    with col3:
        bhk_filter = st.multiselect("BHK", [1, 2, 3, 4, 5], default=[])
        investment_only = st.checkbox("Good Investments Only", value=False)
    
    with col4:
        grade_filter = st.multiselect("Investment Grade", ['A', 'B', 'C', 'D'], default=[])
        sort_by = st.selectbox("Sort By", ['Price (Low to High)', 'Price (High to Low)', 
                                           'Investment Score', 'Size'])
    
    # Apply filters
    filtered = df.copy()
    
    if filter_state:
        filtered = filtered[filtered['State'].isin(filter_state)]
    if filter_type:
        filtered = filtered[filtered['Property_Type'].isin(filter_type)]
    if bhk_filter:
        filtered = filtered[filtered['BHK'].isin(bhk_filter)]
    if grade_filter:
        filtered = filtered[filtered['Investment_Grade'].isin(grade_filter)]
    if investment_only:
        filtered = filtered[filtered['Good_Investment'] == 1]
    
    filtered = filtered[
        (filtered['Price_in_Lakhs'] >= price_range[0]) & 
        (filtered['Price_in_Lakhs'] <= price_range[1]) &
        (filtered['Size_in_SqFt'] >= size_range[0]) &
        (filtered['Size_in_SqFt'] <= size_range[1])
    ]
    
    # Sort
    if sort_by == 'Price (Low to High)':
        filtered = filtered.sort_values('Price_in_Lakhs')
    elif sort_by == 'Price (High to Low)':
        filtered = filtered.sort_values('Price_in_Lakhs', ascending=False)
    elif sort_by == 'Investment Score':
        filtered = filtered.sort_values('Overall_Investment_Score', ascending=False)
    else:
        filtered = filtered.sort_values('Size_in_SqFt', ascending=False)
    
    # Results
    st.markdown(f"**Found {len(filtered):,} properties**")
    
    # Display columns
    display_cols = ['City', 'State', 'Property_Type', 'BHK', 'Size_in_SqFt', 
                   'Price_in_Lakhs', 'Investment_Grade', 'Overall_Investment_Score',
                   'Predicted_Price_5Y', 'Price_Gain_Pct_5Y']
    
    if len(filtered) > 0:
        display_df = filtered[display_cols].head(100).copy()
        display_df.columns = ['City', 'State', 'Type', 'BHK', 'Size (sqft)', 
                             'Price (₹L)', 'Grade', 'Score', '5Y Price (₹L)', '5Y Gain %']
        display_df['Score'] = display_df['Score'].round(1)
        display_df['5Y Price (₹L)'] = display_df['5Y Price (₹L)'].round(2)
        display_df['5Y Gain %'] = display_df['5Y Gain %'].round(1)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Download option
        csv = filtered[display_cols].to_csv(index=False)
        st.download_button("📥 Download Results (CSV)", csv, "filtered_properties.csv", "text/csv")
    else:
        st.warning("No properties match your criteria. Try adjusting the filters.")


if __name__ == "__main__":
    main()
