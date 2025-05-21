import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from collections import Counter
import base64
from io import StringIO
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud  # Add this import
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources (first time only)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set page configuration
st.set_page_config(page_title="Multi-Brand Product Analysis", layout="wide", initial_sidebar_state="expanded")

# Title
st.title("Multi-Brand Product Analysis Dashboard")

# Helper functions for data cleaning and analysis
def clean_data(df):
    """Clean and prepare the dataset for analysis"""
    # Make a copy to avoid modifying original data
    df_clean = df.copy()
    
    # Convert price columns to numeric, removing any non-numeric characters
    price_cols = ['price_amount', 'price_mrp']
    for col in price_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Convert review count and ratings to numeric
    if 'review_count' in df_clean.columns:
        df_clean['review_count'] = pd.to_numeric(df_clean['review_count'], errors='coerce')
    
    if 'ratings' in df_clean.columns:
        df_clean['ratings'] = pd.to_numeric(df_clean['ratings'], errors='coerce')
        
    # Convert delivery_days to numeric
    if 'delivery_days' in df_clean.columns:
        df_clean['delivery_days'] = pd.to_numeric(df_clean['delivery_days'], errors='coerce')
    
    # Standardize brand names (lowercase)
    if 'brand_d2c' in df_clean.columns:
        df_clean['brand_d2c'] = df_clean['brand_d2c'].str.lower()
    
    # Clean up availability status
    if 'availability' in df_clean.columns:
        df_clean['availability'] = df_clean['availability'].str.lower()
        df_clean['availability'] = df_clean['availability'].fillna('unknown')
        df_clean['is_in_stock'] = df_clean['availability'].str.contains('in stock').astype(int)
    
    # Extract domain from product_url (for brand analysis if brand_d2c is missing)
    if 'product_url' in df_clean.columns:
        df_clean['domain'] = df_clean['product_url'].str.extract(r'https?://(?:www\.)?([^/]+)')
    
    # Clean bestseller tag
    if 'bestseller_tag' in df_clean.columns:
        df_clean['bestseller_tag'] = df_clean['bestseller_tag'].fillna('FALSE')
        df_clean['bestseller_tag'] = df_clean['bestseller_tag'].apply(lambda x: True if str(x).upper() == 'TRUE' else False)
    
    return df_clean

def get_price_range(price):
    """Categorize price into ranges"""
    if pd.isna(price):
        return "Unknown"
    if price < 1000:
        return "Budget (< ₹1,000)"
    elif price < 5000:
        return "Affordable (₹1,000 - ₹5,000)"
    elif price < 20000:
        return "Mid-range (₹5,000 - ₹20,000)"
    elif price < 50000:
        return "Premium (₹20,000 - ₹50,000)"
    else:
        return "Luxury (> ₹50,000)"

def extract_dimensions(text):
    """Extract dimensions from text descriptions"""
    if not isinstance(text, str):
        return None
    
    # Look for common dimension patterns (e.g., 5 x 8 inches, 21.5 cm, etc.)
    dimension_patterns = [
        r'(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*(?:x\s*(\d+(?:\.\d+)?))?\s*(?:inches|inch|in|cm|centimeters|mm)',
        r'diameter:?\s*(\d+(?:\.\d+)?)\s*(?:inches|inch|in|cm|centimeters|mm)',
        r'length:?\s*(\d+(?:\.\d+)?)\s*(?:inches|inch|in|cm|centimeters|mm)',
        r'width:?\s*(\d+(?:\.\d+)?)\s*(?:inches|inch|in|cm|centimeters|mm)',
        r'height:?\s*(\d+(?:\.\d+)?)\s*(?:inches|inch|in|cm|centimeters|mm)'
    ]
    
    for pattern in dimension_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    
    return None

# Main function to load and process data
def process_data(df):
    # Clean the data
    df_clean = clean_data(df)
    
    # Add price range category
    if 'price_amount' in df_clean.columns:
        df_clean['price_range'] = df_clean['price_amount'].apply(get_price_range)
    
    # Extract dimensions from description and attributes
    text_columns = ['description', 'attributes_text', 'other_useful_information_text']
    for col in text_columns:
        if col in df_clean.columns:
            df_clean['extracted_dimensions'] = df_clean[col].apply(extract_dimensions)
            # If we found dimensions, break
            if df_clean['extracted_dimensions'].notna().sum() > 0:
                break
    
    # Calculate discount percentage if both MRP and amount are available
    if 'price_mrp' in df_clean.columns and 'price_amount' in df_clean.columns:
        df_clean['discount_percentage'] = ((df_clean['price_mrp'] - df_clean['price_amount']) / 

                                          df_clean['price_mrp'] * 100).round(2)
        # Clean up negative discounts (where amount > MRP)
        df_clean.loc[df_clean['discount_percentage'] < 0, 'discount_percentage'] = 0
    
    return df_clean

# Main app
def main():
    # Sidebar - Upload data
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("Data loaded successfully!")
            
            # Display raw data stats
            st.sidebar.write(f"Total products: {len(df)}")
            
            # Process data
            df_processed = process_data(df)
            
            # Global filters in sidebar
            st.sidebar.header("Global Filters")
            
            # Brand filter
            if 'brand_d2c' in df_processed.columns:
                all_brands = sorted(df_processed['brand_d2c'].dropna().unique().tolist())
                selected_brands = st.sidebar.multiselect("Filter by Brand", all_brands)
                if selected_brands:
                    df_processed = df_processed[df_processed['brand_d2c'].isin(selected_brands)]
            
            # Category filter
            category_cols = [col for col in ['category', 'sub_category'] 

                            if col in df_processed.columns]
            
            if category_cols:
                main_cat_col = category_cols[0]  # Use first available category column
                all_categories = sorted(df_processed[main_cat_col].dropna().unique().tolist())
                selected_categories = st.sidebar.multiselect("Filter by Category", all_categories)
                if selected_categories:
                    df_processed = df_processed[df_processed[main_cat_col].isin(selected_categories)]
            
            # Price range filter
            if 'price_range' in df_processed.columns:
                all_price_ranges = sorted(df_processed['price_range'].dropna().unique().tolist())
                selected_price_ranges = st.sidebar.multiselect("Filter by Price Range", all_price_ranges)
                if selected_price_ranges:
                    df_processed = df_processed[df_processed['price_range'].isin(selected_price_ranges)]
            
            # Material filter
            if 'material' in df_processed.columns:
                all_materials = sorted(df_processed['material'].dropna().unique().tolist())
                selected_materials = st.sidebar.multiselect("Filter by Material", all_materials)
                if selected_materials:
                    df_processed = df_processed[df_processed['material'].isin(selected_materials)]
            
            # Display filtered data count
            st.sidebar.write(f"Filtered products: {len(df_processed)}")
            
            # Reset filters button
            if st.sidebar.button("Reset All Filters"):
                df_processed = process_data(df)
            
            # Sidebar - Analysis Selection
            st.sidebar.header("Choose Analysis")
            analysis_type = st.sidebar.selectbox("Select Analysis", 

                                                ["Overview",
                                                "1. Brand Analysis",
                                                "2. Category Analysis",
                                                "3. Price Analysis",
                                                "4. Material Analysis",
                                                "5. Availability & Logistics",
                                                "6. Customer Reviews",
                                                "7. Text Analysis",
                                                "8. Cross-Analysis"])
            
            # Main Area - Analysis Output
            if analysis_type == "Overview":
                display_overview(df_processed)
            
            elif analysis_type == "1. Brand Analysis":
                # Additional brand-specific filters
                st.subheader("Brand Analysis Filters")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Filter by minimum product count
                    if 'brand_d2c' in df_processed.columns:
                        brand_counts = df_processed['brand_d2c'].value_counts()
                        min_products = st.slider("Minimum Products per Brand", 
                                              min_value=1, 
                                              max_value=int(brand_counts.max()), 
                                              value=1)
                        
                        brands_with_min_products = brand_counts[brand_counts >= min_products].index.tolist()
                        df_brand_filtered = df_processed[df_processed['brand_d2c'].isin(brands_with_min_products)]
                    else:
                        df_brand_filtered = df_processed
                
                with col2:
                    # Filter by D2C status if available
                    if 'brand_d2c' in df_processed.columns:
                        d2c_options = sorted(df_processed['brand_d2c'].dropna().unique().tolist())
                        selected_d2c = st.multiselect("Filter by D2C Status", d2c_options)
                        if selected_d2c:
                            df_brand_filtered = df_brand_filtered[df_brand_filtered['brand_d2c'].isin(selected_d2c)]
                
                display_brand_analysis(df_brand_filtered)
            
            elif analysis_type == "2. Category Analysis":
                # Additional category-specific filters
                st.subheader("Category Analysis Filters")
                
                # If we have subcategory columns
                if len(category_cols) > 1:
                    sub_cat_col = category_cols[1]  # Use second category column
                    all_subcategories = sorted(df_processed[sub_cat_col].dropna().unique().tolist())
                    selected_subcategories = st.multiselect("Filter by Subcategory", all_subcategories)
                    if selected_subcategories:
                        df_cat_filtered = df_processed[df_processed[sub_cat_col].isin(selected_subcategories)]
                    else:
                        df_cat_filtered = df_processed
                else:
                    df_cat_filtered = df_processed
                
                display_category_analysis(df_cat_filtered)
            
            elif analysis_type == "3. Price Analysis":
                # Additional price-specific filters
                st.subheader("Price Analysis Filters")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Price range slider
                    if 'price_amount' in df_processed.columns:
                        min_price = float(df_processed['price_amount'].min())
                        max_price = float(df_processed['price_amount'].max())
                        price_range = st.slider("Price Range (₹)", 



                                              min_value=min_price, 
                                              max_value=max_price, 
                                              value=(min_price, max_price))
                        
                        df_price_filtered = df_processed[(df_processed['price_amount'] >= price_range[0]) & 
                                                      (df_processed['price_amount'] <= price_range[1])]
                    else:
                        df_price_filtered = df_processed
                
                with col2:
                    # Discount filter if available
                    if 'discount_percentage' in df_processed.columns:
                        min_discount = st.number_input("Minimum Discount %", 
                                                    min_value=0.0, 
                                                    max_value=100.0, 
                                                    value=0.0)
                        
                        df_price_filtered = df_price_filtered[df_price_filtered['discount_percentage'] >= min_discount]
                
                display_price_analysis(df_price_filtered)
            
            elif analysis_type == "4. Material Analysis":
                # Additional material-specific filters
                st.subheader("Material Analysis Filters")
                
                # Color filter if available
                if 'color' in df_processed.columns:
                    all_colors = sorted(df_processed['color'].dropna().unique().tolist())
                    selected_colors = st.multiselect("Filter by Color", all_colors)
                    if selected_colors:
                        df_material_filtered = df_processed[df_processed['color'].isin(selected_colors)]
                    else:
                        df_material_filtered = df_processed
                else:
                    df_material_filtered = df_processed
                
                # Aesthetic filter if available
                if 'aesthetic' in df_processed.columns:
                    all_aesthetics = sorted(df_processed['aesthetic'].dropna().unique().tolist())
                    selected_aesthetics = st.multiselect("Filter by Aesthetic", all_aesthetics)
                    if selected_aesthetics:
                        df_material_filtered = df_material_filtered[df_material_filtered['aesthetic'].isin(selected_aesthetics)]
                
                display_material_analysis(df_material_filtered)
            
            elif analysis_type == "5. Availability & Logistics":
                # Additional availability-specific filters
                st.subheader("Availability & Logistics Filters")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Stock status filter
                    if 'availability' in df_processed.columns:
                        all_availability = sorted(df_processed['availability'].dropna().unique().tolist())
                        selected_availability = st.multiselect("Filter by Availability Status", all_availability)
                        if selected_availability:
                            df_avail_filtered = df_processed[df_processed['availability'].isin(selected_availability)]
                        else:
                            df_avail_filtered = df_processed
                    else:
                        df_avail_filtered = df_processed
                
                with col2:
                    # Delivery days filter
                    if 'delivery_days' in df_processed.columns:
                        max_delivery = int(df_processed['delivery_days'].max())
                        delivery_days = st.slider("Maximum Delivery Days", 



                                                min_value=1, 
                                                max_value=max_delivery, 
                                                value=max_delivery)
                        
                        df_avail_filtered = df_avail_filtered[df_avail_filtered['delivery_days'] <= delivery_days]
                
                display_availability_analysis(df_avail_filtered)
            
            elif analysis_type == "6. Customer Reviews":
                # Additional review-specific filters
                st.subheader("Customer Review Filters")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Rating filter
                    if 'ratings' in df_processed.columns:
                        min_rating = st.slider("Minimum Rating", 
                                            min_value=0.0, 
                                            max_value=5.0, 
                                            value=0.0, 
                                            step=0.5)
                        
                        df_review_filtered = df_processed[df_processed['ratings'] >= min_rating]
                    else:
                        df_review_filtered = df_processed
                
                with col2:
                    # Review count filter
                    if 'review_count' in df_processed.columns:
                        min_reviews = st.number_input("Minimum Review Count", 


                                                    min_value=0, 
                                                    value=0)
                        
                        df_review_filtered = df_review_filtered[df_review_filtered['review_count'] >= min_reviews]
                
                display_review_analysis(df_review_filtered)
            
            elif analysis_type == "7. Text Analysis":
                # Text analysis doesn't need additional filters as it has its own selection mechanism
                display_text_analysis(df_processed)
            
            elif analysis_type == "8. Cross-Analysis":
                # Cross-analysis uses the globally filtered dataset
                display_cross_analysis(df_processed)
        
        except Exception as e:
            st.error(f"Error: {e}")
            st.error("Please check your CSV file format and try again.")
    else:
        st.info("Please upload your CSV file to start the analysis.")
        st.write("This app analyzes multi-brand product data with columns like product_name, brand_d2c, price_amount, category, etc.")

def display_overview(df):
    st.header("Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Products", len(df))
        
        # Brand count
        if 'brand_d2c' in df.columns:
            brand_count = df['brand_d2c'].nunique()
            st.metric("Number of Brands", brand_count)
        
        # Category count
        if 'category' in df.columns:
            category_count = df['category'].nunique()
            st.metric("Number of Main Categories", category_count)
    
    with col2:
        # Price range
        if 'price_amount' in df.columns:
            min_price = df['price_amount'].min()
            max_price = df['price_amount'].max()
            avg_price = df['price_amount'].mean()
            
            st.metric("Price Range", f"₹{min_price:,.2f} - ₹{max_price:,.2f}")
            st.metric("Average Price", f"₹{avg_price:,.2f}")
    
    # Sample data preview
    st.subheader("Data Sample")
    st.dataframe(df.head())
    
    # Missing value analysis
    st.subheader("Missing Values Analysis")
    missing_data = df.isnull().sum().reset_index()
    missing_data.columns = ['Column', 'Missing Count']
    missing_data['Missing Percentage'] = (missing_data['Missing Count'] / len(df)) * 100
    missing_data = missing_data.sort_values('Missing Percentage', ascending=False)
    
    fig = px.bar(missing_data[missing_data['Missing Count'] > 0], 




                  x='Column', y='Missing Percentage', 
                  title="Missing Values by Column (%)",
                  color='Missing Percentage',
                  color_continuous_scale='Reds')
    st.plotly_chart(fig)

def display_brand_analysis(df):
    st.header("Brand Analysis")
    
    if 'brand_d2c' not in df.columns:
        st.warning("Brand name column not found in the dataset")
        return
        
    # Brand distribution
    st.subheader("Brand Distribution")
    brand_counts = df['brand_d2c'].value_counts().reset_index()
    brand_counts.columns = ['Brand', 'Count']
    
    fig = px.bar(brand_counts, x='Brand', y='Count', 

                  color='Count', color_continuous_scale='Blues')
    st.plotly_chart(fig)
    
    # Brand pricing
    if 'price_amount' in df.columns:
        st.subheader("Brand Pricing Analysis")
        
        # Get top brands by count for analysis
        top_brands = df['brand_d2c'].value_counts().index.tolist()
        df_top_brands = df[df['brand_d2c'].isin(top_brands)]
        
        # Create a box plot showing price distribution by brand
        fig = px.box(
            df_top_brands,
            x='brand_d2c',
            y='price_amount',
            title="Brand Pricing Analysis: Price Distribution by Brand",
            color='brand_d2c',
            labels={
                'brand_d2c': 'Brand',
                'price_amount': 'Price (₹)'
            },
            height=600
        )
        
        # Improve layout
        fig.update_layout(
            xaxis={'categoryorder': 'total descending'},
            showlegend=False,  # Hide legend as colors are just for visual distinction
            margin=dict(l=50, r=50, t=80, b=150)
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig)
        
        # Optionally, add a table with summary statistics
        st.subheader("Brand Pricing Summary Statistics")
        
        brand_price = df.groupby('brand_d2c')['price_amount'].agg([
            'mean', 'median', 'min', 'max','count'
        ]).reset_index()
        brand_price.columns = [
            'brand_d2c', 'Average Price', 'Median Price', 'Min Price', 
            'Max Price', 'Product Count'
        ]
        brand_price = brand_price[brand_price['brand_d2c'].isin(top_brands)].sort_values('Product Count', ascending=False)
        
        # Round the Average Price to 2 decimal places
        brand_price['Average Price'] = brand_price['Average Price'].round(2)

        # If you want to round other price columns as well
        brand_price['Median Price'] = brand_price['Median Price'].round(2)
        brand_price['Min Price'] = brand_price['Min Price'].round(2)
        brand_price['Max Price'] = brand_price['Max Price'].round(2)
        
        st.dataframe(brand_price)
    
    # Brand category focus
    if 'category' in df.columns:
        st.subheader("Brand Category Focus")
        
        # Get top brands and categories
        top_brands = df['brand_d2c'].value_counts().index.tolist()
        df_top_brands = df[df['brand_d2c'].isin(top_brands)]
        main_cat_col = df['category']

        # Create long-format data for bubble chart
        brand_cat_counts = df_top_brands.groupby(['brand_d2c', main_cat_col]).size().reset_index()
        brand_cat_counts.columns = ['Brand', 'Category', 'Count']

        # Create bubble chart
        fig = px.scatter(
            brand_cat_counts,
            x='Brand',
            y='Category',
            size='Count',
            color='Brand',
            title="Brand Category Focus: Bubble Chart",
            size_max=60,
            height=700
        )
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig)
    
    # D2C analysis
    if 'brand_d2c' in df.columns:
        st.subheader("Direct-to-Consumer (D2C) Analysis")
        
        d2c_counts = df['brand_d2c'].value_counts().reset_index()
        d2c_counts.columns = ['D2C Status', 'Count']
        
        fig = px.pie(d2c_counts, values='Count', names='D2C Status', 

                    title="Distribution of D2C vs Non-D2C Brands")
        st.plotly_chart(fig)

def display_category_analysis(df):
    st.header("Product Category Analysis")
    
    # Check for category columns
    cat_columns = [col for col in ['category','sub_category'] 

                  if col in df.columns]
    
    if not cat_columns:
        st.warning("No category columns found in the dataset")
        return
    
    # Category distribution
    st.subheader("Main Category Distribution")
    
    # Use the first available category column
    main_cat_col = cat_columns[0] 
    
    cat_counts = df[main_cat_col].value_counts().reset_index()
    cat_counts.columns = ['Category', 'Count']
    
    fig = px.bar(cat_counts.head(15), x='Category', y='Count', 


                  title=f"Top 15 {main_cat_col} by Product Count",
                  color='Count', color_continuous_scale='Greens')
    st.plotly_chart(fig)
    
    # Category hierarch
    
    # Price range by category
    if 'price_amount' in df.columns and main_cat_col in df.columns:
        st.subheader("Price Range by Category")
        
        cat_price = df.groupby(main_cat_col)['price_amount'].agg(['mean', 'min', 'max', 'count']).reset_index()
        cat_price.columns = ['Category', 'Average Price', 'Min Price', 'Max Price', 'Product Count']
        cat_price = cat_price.sort_values('Average Price', ascending=False)
        
        fig = px.bar(cat_price.head(15), x='Category', y='Average Price', 
                    error_y='Max Price', error_y_minus='Min Price',
                    title=f"Average Price by {main_cat_col} (with Min-Max Range)")
        st.plotly_chart(fig)
        
        # Box plot to show price distribution within categories
        top_cats = df[main_cat_col].value_counts().head(8).index.tolist()
        fig = px.box(df[df[main_cat_col].isin(top_cats)], 
                    x=main_cat_col, y='price_amount',
                    title=f"Price Distribution by Top {main_cat_col}")
        st.plotly_chart(fig)
    
    # Brand-Category competition
    if 'brand_d2c' in df.columns and main_cat_col in df.columns:
        st.subheader("Cross-Brand Category Competition")
        
        # Get top categories and brands
        top_cats = df[main_cat_col].value_counts().head(10).index.tolist()
        top_brands = df['brand_d2c'].value_counts().head(8).index.tolist()
        
        # Create a heatmap of brands vs categories
        brand_cat_matrix = pd.crosstab(df[df['brand_d2c'].isin(top_brands)]['brand_d2c'], 

                                      df[df['brand_d2c'].isin(top_brands)][main_cat_col])
        
        # Only include top categories
        brand_cat_matrix = brand_cat_matrix[[col for col in brand_cat_matrix.columns if col in top_cats]]
        
        # Plot heatmap
        fig = px.imshow(brand_cat_matrix, 



                        title=f"Brand Competition by {main_cat_col}",
                        labels=dict(x=main_cat_col, y="Brand", color="Product Count"),
                        color_continuous_scale='Viridis')
        st.plotly_chart(fig)

def display_price_analysis(df):
    st.header("Price Analysis")
    
    if 'price_amount' not in df.columns:
        st.warning("Price amount column not found in the dataset")
        return
    
    # Price distribution
    st.subheader("Price Distribution")
    
    fig = px.histogram(df, x='price_amount', nbins=50,



                      title="Price Distribution Histogram",
                      labels={'price_amount': 'Price (₹)'},
                      marginal='box')
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig)
    
    # Price segments
    st.subheader("Price Segments")
    
    if 'price_range' in df.columns:
        price_range_counts = df['price_range'].value_counts().reset_index()
        price_range_counts.columns = ['Price Range', 'Count']
        
        fig = px.pie(price_range_counts, values='Count', names='Price Range', 

                    title="Distribution by Price Range")
        st.plotly_chart(fig)
    
    # Price vs. material
    if 'material' in df.columns:
        st.subheader("Price vs. Material")
        
        # Get top materials by count
        top_materials = df['material'].value_counts().head(8).index.tolist()
        
        # Filter df for top materials
        df_top_materials = df[df['material'].isin(top_materials)]
        
        fig = px.box(df_top_materials, x='material', y='price_amount',


                    title="Price Distribution by Top Materials",
                    labels={'material': 'Material', 'price_amount': 'Price (₹)'})
        st.plotly_chart(fig)
    
    # Discount analysis
    if 'discount_percentage' in df.columns:
        st.subheader("Discount Analysis")
        
        # Filter out zero discounts
        df_with_discount = df[df['discount_percentage'] > 0]
        
        if len(df_with_discount) > 0:
            fig = px.histogram(df_with_discount, x='discount_percentage', nbins=20,


                              title="Discount Percentage Distribution",
                              labels={'discount_percentage': 'Discount %'})
            st.plotly_chart(fig)
            
            # Average discount by brand
            if 'brand_d2c' in df.columns:
                brand_discount = df_with_discount.groupby('brand_d2c')['discount_percentage'].mean().reset_index()
                brand_discount.columns = ['Brand', 'Average Discount %']
                brand_discount = brand_discount.sort_values('Average Discount %', ascending=False)
                
                fig = px.bar(brand_discount.head(15), x='Brand', y='Average Discount %',


                            title="Average Discount Percentage by Brand",
                            color='Average Discount %', color_continuous_scale='Reds')
                st.plotly_chart(fig)
        else:
            st.info("No products with discounts found in the dataset")
    
    # Price trend by category
    if 'category' in df.columns:
        st.subheader("Price Trend by Category")
        
        cat_price_trend = df.groupby('category')['price_amount'].mean().reset_index()
        cat_price_trend.columns = ['Category', 'Average Price']
        cat_price_trend = cat_price_trend.sort_values('Average Price', ascending=False)
        
        fig = px.bar(cat_price_trend.head(15), x='Category', y='Average Price',


                    title="Average Price by Category",
                    color='Average Price', color_continuous_scale='Blues')
        st.plotly_chart(fig)

def display_material_analysis(df):
    st.header("Material & Attributes Analysis")
    
    # Material distribution
    if 'material' in df.columns:
        st.subheader("Material Distribution")
        
        material_counts = df['material'].value_counts().reset_index()
        material_counts.columns = ['Material', 'Count']
        
        fig = px.bar(material_counts.head(15), x='Material', y='Count',


                    title="Top 15 Materials by Product Count",
                    color='Count', color_continuous_scale='Purples')
        st.plotly_chart(fig)
        
        # Material by brand
        if 'brand_d2c' in df.columns:
            st.subheader("Material by Brand")
            
            # Get top brands and materials
            top_brands = df['brand_d2c'].value_counts().head(8).index.tolist()
            top_materials = df['material'].value_counts().head(8).index.tolist()
            
            # Create filtered DataFrame
            df_filtered = df[df['brand_d2c'].isin(top_brands) & df['material'].isin(top_materials)]
            
            # Create a heatmap
            brand_material_matrix = pd.crosstab(df_filtered['brand_d2c'], df_filtered['material'])
            
            fig = px.imshow(brand_material_matrix,
                          title="Brand vs Material Heatmap",
                          labels=dict(x="Material", y="Brand", color="Product Count"),
                          color_continuous_scale='Viridis')
            st.plotly_chart(fig)
    
    # Color analysis
    if 'color' in df.columns:
        st.subheader("Color Analysis")
        
        color_counts = df['color'].value_counts().reset_index()
        color_counts.columns = ['Color', 'Count']
        
        fig = px.bar(color_counts.head(15), x='Color', y='Count',


                    title="Top 15 Colors by Product Count",
                    color='Count', color_continuous_scale='Oranges')
        st.plotly_chart(fig)
    
    # Finish type analysis
    if 'finish' in df.columns:
        st.subheader("Finish Type Analysis")
        
        finish_counts = df['finish'].value_counts().reset_index()
        finish_counts.columns = ['Finish', 'Count']
        
        fig = px.pie(finish_counts.head(10), values='Count', names='Finish',

                    title="Top 10 Finish Types Distribution")
        st.plotly_chart(fig)
    
    # Aesthetic type analysis
    if 'aesthetic' in df.columns:
        st.subheader("Aesthetic Style Analysis")
        
        aesthetic_counts = df['aesthetic'].value_counts().reset_index()
        aesthetic_counts.columns = ['Aesthetic', 'Count']
        
        fig = px.bar(aesthetic_counts.head(15), x='Aesthetic', y='Count',


                    title="Top 15 Aesthetic Styles by Product Count",
                    color='Count', color_continuous_scale='Greens')
        st.plotly_chart(fig)
    
    # Dimension analysis
    if 'extracted_dimensions' in df.columns:
        st.subheader("Product Dimensions Analysis")
        
        # Count products with dimension information
        dim_count = df['extracted_dimensions'].notna().sum()
        no_dim_count = df['extracted_dimensions'].isna().sum()
        
        dim_data = pd.DataFrame({
            'Status': ['With Dimensions', 'Without Dimensions'],
            'Count': [dim_count, no_dim_count]
        })
        
        fig = px.pie(dim_data, values='Count', names='Status',

                    title="Products With/Without Dimension Information")
        st.plotly_chart(fig)

def display_availability_analysis(df):
    st.header("Availability & Logistics Analysis")
    
    # Stock status
    if 'availability' in df.columns:
        st.subheader("Stock Status")
        
        avail_counts = df['availability'].value_counts().reset_index()
        avail_counts.columns = ['Status', 'Count']
        
        fig = px.pie(avail_counts, values='Count', names='Status',

                    title="Product Availability Status")
        st.plotly_chart(fig)
        
        # Stock status by brand
        if 'brand_d2c' in df.columns and 'is_in_stock' in df.columns:
            brand_stock = df.groupby('brand_d2c')['is_in_stock'].mean().reset_index()
            brand_stock.columns = ['Brand', 'In-Stock Percentage']
            brand_stock['In-Stock Percentage'] = brand_stock['In-Stock Percentage'] * 100
            brand_stock = brand_stock.sort_values('In-Stock Percentage')
            
            fig = px.bar(brand_stock.head(15), x='Brand', y='In-Stock Percentage',


                        title="In-Stock Percentage by Brand",
                        color='In-Stock Percentage', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig)
    
    # Delivery time analysis
    if 'delivery_days' in df.columns:
        st.subheader("Delivery Time Analysis")
        
        # Histogram of delivery days
        fig = px.histogram(df, x='delivery_days', nbins=20,


                          title="Distribution of Delivery Days",
                          labels={'delivery_days': 'Delivery Days'})
        st.plotly_chart(fig)
        
        # Average delivery days by brand
        if 'brand_d2c' in df.columns:
            brand_delivery = df.groupby('brand_d2c')['delivery_days'].mean().reset_index()
            brand_delivery.columns = ['Brand', 'Average Delivery Days']
            brand_delivery = brand_delivery.sort_values('Average Delivery Days')
            
            fig = px.bar(brand_delivery.head(15), x='Brand', y='Average Delivery Days',


                        title="Average Delivery Days by Brand",
                        color='Average Delivery Days', color_continuous_scale='Blues_r')
            st.plotly_chart(fig)
    
    # Bestseller analysis
    if 'bestseller_tag' in df.columns:
        st.subheader("Bestseller Analysis")
        
        bestseller_counts = df['bestseller_tag'].value_counts().reset_index()
        bestseller_counts.columns = ['Bestseller', 'Count']
        
        fig = px.pie(bestseller_counts, values='Count', names='Bestseller',

                    title="Bestseller Tag Distribution")
        st.plotly_chart(fig)
        
        # Compare bestseller prices
        if 'price_amount' in df.columns:
            bestseller_price = df.groupby('bestseller_tag')['price_amount'].mean().reset_index()
            bestseller_price.columns = ['Bestseller', 'Average Price']
            
            fig = px.bar(bestseller_price, x='Bestseller', y='Average Price',


                        title="Average Price: Bestsellers vs Non-Bestsellers",
                        color='Bestseller', color_discrete_sequence=['#ff7f0e', '#1f77b4'])
            st.plotly_chart(fig)
    
    # Return policy analysis
    if 'return_policy' in df.columns:
        st.subheader("Return Policy Analysis")
        
        # Extract key return policy types
        df['return_type'] = df['return_policy'].apply(lambda x: 



                                                    'No Return' if isinstance(x, str) and 'non-returnable' in x.lower() else
                                                    'Free Shipping' if isinstance(x, str) and 'free' in x.lower() and 'shipping' in x.lower() else
                                                    'Returns Allowed' if isinstance(x, str) and len(x) > 3 else 'Unknown')
        
        return_type_counts = df['return_type'].value_counts().reset_index()
        return_type_counts.columns = ['Return Type', 'Count']
        
        fig = px.pie(return_type_counts, values='Count', names='Return Type',

                    title="Return Policy Types")
        st.plotly_chart(fig)
    
    # Manufacturing country
    if 'manufacturing_country' in df.columns:
        st.subheader("Manufacturing Country Analysis")
        
        country_counts = df['manufacturing_country'].value_counts().reset_index()
        country_counts.columns = ['Country', 'Count']
        
        fig = px.bar(country_counts.head(15), x='Country', y='Count',


                    title="Product Count by Manufacturing Country",
                    color='Count', color_continuous_scale='Viridis')
        st.plotly_chart(fig)

def display_review_analysis(df):
    st.header("Customer Review Analysis")
    
    # Check if we have review data
    review_cols = ['ratings', 'review_count', 'reviews', 'positive_reviews', 'negative_reviews']
    has_review_data = any(col in df.columns for col in review_cols)
    
    if not has_review_data:
        st.warning("No review data found in the dataset")
        return
    
    # Rating distribution
    if 'ratings' in df.columns:
        st.subheader("Rating Distribution")
        
        # Filter out null ratings
        df_with_ratings = df[df['ratings'].notna()]
        
        if len(df_with_ratings) > 0:
            fig = px.histogram(df_with_ratings, x='ratings', nbins=10,



                              title="Distribution of Product Ratings",
                              labels={'ratings': 'Rating'},
                              color_discrete_sequence=['#1f77b4'])
            st.plotly_chart(fig)
            
            # Average rating by brand
            if 'brand_d2c' in df.columns:
                st.subheader("Average Rating by Brand")
                
                brand_ratings = df_with_ratings.groupby('brand_d2c')['ratings'].agg(['mean', 'count']).reset_index()
                brand_ratings.columns = ['Brand', 'Average Rating', 'Review Count']
                brand_ratings = brand_ratings[brand_ratings['Review Count'] >= 5]  # Only brands with at least 5 reviews
                brand_ratings = brand_ratings.sort_values('Average Rating', ascending=False)
                
                fig = px.bar(brand_ratings.head(15), x='Brand', y='Average Rating',



                            title="Top 15 Brands by Average Rating (min 5 reviews)",
                            color='Average Rating', color_continuous_scale='Blues',
                            text_auto='.2f')
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig)
    
    # Review count analysis
    if 'review_count' in df.columns:
        st.subheader("Review Count Analysis")
        
        # Filter out null review counts
        df_with_reviews = df[df['review_count'].notna() & (df['review_count'] > 0)]
        
        if len(df_with_reviews) > 0:
            fig = px.histogram(df_with_reviews, x='review_count', nbins=20, log_y=True,


                              title="Distribution of Review Counts (log scale)",
                              labels={'review_count': 'Number of Reviews'})
            st.plotly_chart(fig)
            
            # Top products by review count
            st.subheader("Most Reviewed Products")
            
            top_reviewed = df_with_reviews.sort_values('review_count', ascending=False).head(10)
            





            fig = px.bar(top_reviewed, x='review_count', y='name',
                        title="Top 10 Most Reviewed Products",
                        labels={'review_count': 'Number of Reviews', 'name': 'Product'},
                        color='review_count', color_continuous_scale='Viridis',
                        orientation='h')
            st.plotly_chart(fig)
    
    # Positive vs Negative reviews analysis
    if 'positive_reviews' in df.columns and 'negative_reviews' in df.columns:
        st.subheader("Positive vs Negative Reviews Analysis")
        
        # Convert to string and check if not empty
        df['has_positive'] = df['positive_reviews'].astype(str).apply(lambda x: 0 if x.lower() in ['nan', 'none', ''] else 1)
        df['has_negative'] = df['negative_reviews'].astype(str).apply(lambda x: 0 if x.lower() in ['nan', 'none', ''] else 1)
        
        # Count products with different review types
        review_types = {
            'Both Positive & Negative': sum((df['has_positive'] == 1) & (df['has_negative'] == 1)),
            'Only Positive': sum((df['has_positive'] == 1) & (df['has_negative'] == 0)),
            'Only Negative': sum((df['has_positive'] == 0) & (df['has_negative'] == 1)),
            'No Reviews': sum((df['has_positive'] == 0) & (df['has_negative'] == 0))
        }
        
        review_df = pd.DataFrame({
            'Review Type': list(review_types.keys()),
            'Count': list(review_types.values())
        })
        
        fig = px.pie(review_df, values='Count', names='Review Type',


                    title="Distribution of Review Types",
                    color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig)
    
    # Rating vs Price analysis
    if 'ratings' in df.columns and 'price_amount' in df.columns:
        st.subheader("Rating vs Price Analysis")
        
        # Filter out nulls
        df_rating_price = df[(df['ratings'].notna()) & (df['price_amount'].notna())]
        
        if len(df_rating_price) > 0:
            fig = px.scatter(df_rating_price, x='ratings', y='price_amount',





                            title="Relationship Between Ratings and Price",
                            labels={'ratings': 'Rating', 'price_amount': 'Price'},
                            color='brand_d2c' if 'brand_d2c' in df.columns else None,
                            size='review_count' if 'review_count' in df.columns else None,
                            hover_data=['name'])
            
            # Add trendline
            fig.update_layout(
                shapes=[{
                    'type': 'line',
                    'x0': df_rating_price['ratings'].min(),
                    'y0': df_rating_price['price_amount'].min(),
                    'x1': df_rating_price['ratings'].max(),
                    'y1': df_rating_price['price_amount'].max(),
                    'line': {
                        'color': 'red',
                        'width': 2,
                        'dash': 'dash'
                    }
                }]
            )
            st.plotly_chart(fig)

def display_text_analysis(df):
    st.header("Text Analysis")
    
    # Check if we have text data
    text_cols = ['description', 'attributes_text', 'other_useful_information_text', 'product_description_model_based']
    available_text_cols = [col for col in text_cols if col in df.columns]
    
    if not available_text_cols:
        st.warning("No text data found in the dataset")
        return
    
    # Let user select which text column to analyze
    text_col = st.selectbox("Select text field to analyze", available_text_cols)
    
    # Filter out rows with empty text
    df_with_text = df[df[text_col].notna() & (df[text_col].str.len() > 0)]
    
    if len(df_with_text) == 0:
        st.warning(f"No data found in the {text_col} column")
        return
    
    # Word frequency analysis
    st.subheader("Word Frequency Analysis")
    
    # Combine all text
    all_text = ' '.join(df_with_text[text_col].astype(str))
    
    # Tokenize and clean
    words = re.findall(r'\b[a-zA-Z]{3,15}\b', all_text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    # Count word frequencies
    word_freq = Counter(filtered_words).most_common(30)
    word_freq_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
    
    # Plot word frequencies
    fig = px.bar(word_freq_df, x='Word', y='Frequency',


                title=f"Top 30 Words in {text_col}",
                color='Frequency', color_continuous_scale='Viridis')
    st.plotly_chart(fig)
    
    # Word cloud visualization
    st.subheader("Word Cloud Visualization")
    
    # Create word cloud
    wordcloud_data = ' '.join(filtered_words)
    
    # Generate word cloud image
    plt.figure(figsize=(10, 6))
    wordcloud = WordCloud(width=800, height=400, background_color='white', 

                          max_words=100, contour_width=3, contour_color='steelblue').generate(wordcloud_data)
    
    # Display the word cloud
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    

def display_cross_analysis(df):
    st.header("Cross-Analysis")
    
    # Price vs. Rating by Brand
    if all(col in df.columns for col in ['price_amount', 'ratings', 'brand_d2c']):
        st.subheader("Price vs. Rating by Brand")
        
        # Filter out nulls
        df_filtered = df[(df['price_amount'].notna()) & (df['ratings'].notna())]
        
        if len(df_filtered) > 0:
            # Get top brands by count
            top_brands = df_filtered['brand_d2c'].value_counts().head(10).index.tolist()
            df_top_brands = df_filtered[df_filtered['brand_d2c'].isin(top_brands)]
            
            fig = px.scatter(df_top_brands, x='price_amount', y='ratings',




                            color='brand_d2c', size='review_count' if 'review_count' in df.columns else None,
                            hover_data=['name'],
                            title="Price vs. Rating for Top 10 Brands",
                            labels={'price_amount': 'Price', 'ratings': 'Rating', 'brand_d2c': 'Brand'})
            st.plotly_chart(fig)
    
    # Material vs. Category Analysis
    if all(col in df.columns for col in ['material', 'category']):
        st.subheader("Material vs. Category Analysis")
        
        # Get top materials and categories
        top_materials = df['material'].value_counts().head(10).index.tolist()
        top_categories = df['category'].value_counts().head(10).index.tolist()
        
        # Create cross-tabulation
        df_filtered = df[df['material'].isin(top_materials) & df['category'].isin(top_categories)]
        cross_tab = pd.crosstab(df_filtered['material'], df_filtered['category'])
        
        # Create heatmap
        fig = px.imshow(cross_tab, 



                        labels=dict(x="Category", y="Material", color="Count"),
                        title="Material Usage Across Categories",
                        color_continuous_scale='Viridis')
        st.plotly_chart(fig)
    
    # Price Range vs. Availability
    if all(col in df.columns for col in ['price_range', 'availability']):
        st.subheader("Price Range vs. Availability")
        
        # Create cross-tabulation
        cross_tab = pd.crosstab(df['price_range'], df['availability'])
        
        # Convert to percentage
        cross_tab_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100
        
        # Create heatmap
        fig = px.imshow(cross_tab_pct, 




                        labels=dict(x="Availability", y="Price Range", color="Percentage"),
                        title="Availability Percentage by Price Range",
                        color_continuous_scale='RdYlGn',
                        text_auto='.1f')
        st.plotly_chart(fig)
    
    # Delivery Days vs. Price Range
    if all(col in df.columns for col in ['delivery_days', 'price_range']):
        st.subheader("Delivery Days vs. Price Range")
        
        # Filter out nulls
        df_filtered = df[df['delivery_days'].notna()]
        
        if len(df_filtered) > 0:
            # Calculate average delivery days by price range
            delivery_by_price = df_filtered.groupby('price_range')['delivery_days'].mean().reset_index()
            delivery_by_price.columns = ['Price Range', 'Average Delivery Days']
            
            # Sort by price range (custom order)
            price_order = ["Budget (< ₹1,000)", "Affordable (₹1,000 - ₹5,000)", 


                          "Mid-range (₹5,000 - ₹20,000)", "Premium (₹20,000 - ₹50,000)", 
                          "Luxury (> ₹50,000)", "Unknown"]
            
            delivery_by_price['Price Range'] = pd.Categorical(
                delivery_by_price['Price Range'], 
                categories=price_order, 
                ordered=True
            )
            delivery_by_price = delivery_by_price.sort_values('Price Range')
            
            fig = px.bar(delivery_by_price, x='Price Range', y='Average Delivery Days',


                        title="Average Delivery Days by Price Range",
                        color='Average Delivery Days', color_continuous_scale='Blues')
            st.plotly_chart(fig)
    
    # Brand Performance Analysis
    if all(col in df.columns for col in ['brand_d2c', 'price_amount', 'ratings']):
        st.subheader("Brand Performance Analysis")
        
        # Filter out nulls
        df_filtered = df[(df['price_amount'].notna()) & (df['ratings'].notna())]
        
        if len(df_filtered) > 0:
            # Get top brands by count
            top_brands = df_filtered['brand_d2c'].value_counts().head(15).index.tolist()
            
            # Calculate metrics by brand
            brand_metrics = df_filtered[df_filtered['brand_d2c'].isin(top_brands)].groupby('brand_d2c').agg({
                'price_amount': 'mean',
                'ratings': 'mean',
                'name': 'count'
            }).reset_index()
            
            brand_metrics.columns = ['Brand', 'Average Price', 'Average Rating', 'Product Count']
            
            # Create bubble chart
            fig = px.scatter(brand_metrics, x='Average Price', y='Average Rating',




                            size='Product Count', color='Brand',
                            title="Brand Performance: Price vs. Rating vs. Product Count",
                            labels={'Average Price': 'Average Price (₹)', 'Average Rating': 'Average Rating (0-5)'},
                            hover_data=['Brand', 'Average Price', 'Average Rating', 'Product Count'])
            
            # Add quadrant lines
            avg_price = brand_metrics['Average Price'].mean()
            avg_rating = brand_metrics['Average Rating'].mean()
            
            fig.add_shape(type="line", x0=avg_price, y0=0, x1=avg_price, y1=5,
                        line=dict(color="Red", width=1, dash="dash"))
            
            fig.add_shape(type="line", x0=0, y0=avg_rating, x1=brand_metrics['Average Price'].max(),
                        y1=avg_rating, line=dict(color="Red", width=1, dash="dash"))
            
            # Add annotations for quadrants
            fig.add_annotation(x=avg_price/2, y=avg_rating*1.1, text="Low Price, High Rating",

                              showarrow=False, font=dict(size=10))
            
            fig.add_annotation(x=avg_price*1.5, y=avg_rating*1.1, text="High Price, High Rating",

                              showarrow=False, font=dict(size=10))
            
            fig.add_annotation(x=avg_price/2, y=avg_rating*0.9, text="Low Price, Low Rating",

                              showarrow=False, font=dict(size=10))
            
            fig.add_annotation(x=avg_price*1.5, y=avg_rating*0.9, text="High Price, Low Rating",

                              showarrow=False, font=dict(size=10))
            
            st.plotly_chart(fig)
    
    # Category vs. Price vs. Rating
    if all(col in df.columns for col in ['category', 'price_amount', 'ratings']):
        st.subheader("Category Performance Analysis")
        
        # Filter out nulls
        df_filtered = df[(df['price_amount'].notna()) & (df['ratings'].notna())]
        
        if len(df_filtered) > 0:
            # Get top categories by count
            top_categories = df_filtered['category'].value_counts().head(10).index.tolist()
            
            # Calculate metrics by category
            cat_metrics = df_filtered[df_filtered['category'].isin(top_categories)].groupby('category').agg({
                'price_amount': 'mean',
                'ratings': 'mean',
                'name': 'count'
            }).reset_index()
            
            cat_metrics.columns = ['Category', 'Average Price', 'Average Rating', 'Product Count']
            
            # Create bubble chart
            fig = px.scatter(cat_metrics, x='Average Price', y='Average Rating',




                            size='Product Count', color='Category',
                            title="Category Performance: Price vs. Rating vs. Product Count",
                            labels={'Average Price': 'Average Price (₹)', 'Average Rating': 'Average Rating (0-5)'},
                            hover_data=['Category', 'Average Price', 'Average Rating', 'Product Count'])
            
            st.plotly_chart(fig)

# Call the main function
if __name__ == "__main__":
    main()
