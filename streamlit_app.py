import wxee
import ee
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import model_from_json
import streamlit as st
import leafmap
from google.oauth2 import service_account
from ee import oauth

"# LST Downscaling"
def get_auth():
    service_account_keys=st.secrets['ee_keys']
    credentials=service_account.Credentials.from_service_account_info(service_account_keys,scopes=oauth.SCOPES)
    ee.Initialize(credentials)
    return 'Successfully sync to GEE'
    
get_auth()    

# st.set_page_config(layout="wide")


# Initialize global variables
targetProjection = ee.Projection('EPSG:32643')
ERA5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
DEM = ee.Image("USGS/SRTMGL1_003")
Modis = ee.ImageCollection("MODIS/061/MYD11A1")
MODIS_Ref_250 = ee.ImageCollection("MODIS/061/MYD09GQ")
MODIS_Ref_500 = ee.ImageCollection("MODIS/061/MYD09GA")
L8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
L9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")

# Define Landsat bands
L89_Bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10', 'QA_PIXEL']

scaler_X = joblib.load(r"ANN_72_scaler_X.pkl")
scaler_y = joblib.load(r"ANN_72_scaler_y.pkl")

with open(r"ANN_72_Model_Arc.json", "r") as json_file:
    loaded_model_json = json_file.read()

best_model = model_from_json(loaded_model_json)
best_model.load_weights(r"ANN_72_Model_Weights.h5")

bands=['DOY','Elevation', 'SR_B4','sur_refl_b07', 'SSRDH', 'NDVI','NDBI','LST_Day_1km']


def LandsatUpscale(img):
    return img.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024).reproject(crs=targetProjection, scale=100)

def downsampledLST(img, clip_roi):
    return img.resample('bilinear').reproject(crs=targetProjection, scale=100).clip(clip_roi)

def NDVI_NDBI_NDWI(img):
    ndvi = img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    ndbi = img.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
    ndwi = img.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')
    return img.addBands([ndvi, ndbi, ndwi])

def cloudMask(img):
    qaBand = img.select(['QA_PIXEL'])
    cloudBitMask = 1 << 3  # Bit 3 represents cloud
    cloudShadowBitMask = 1 << 4  # Bit 4 represents cloud shadow
    cloudMask = qaBand.bitwiseAnd(cloudBitMask).neq(0)
    cloudShadowMask = qaBand.bitwiseAnd(cloudShadowBitMask).neq(0)
    combinedMask = cloudMask.Or(cloudShadowMask)
    return img.updateMask(combinedMask.Not())

def applyScaleFactors(image):
    opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
    return image.addBands(opticalBands, None, True).addBands(thermalBands, None, True).select(['SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10']).set('Landsat_Time', ee.Date(image.get('system:time_start')).format('YYYY-MM-dd HH:mm'))

def addBandsToModis(img):
    thermalBands = img.select('LST_Day_1km').multiply(0.02)
    opticalBands = img.select('sur_refl_b.*').multiply(0.0001)
    return img.addBands([thermalBands, opticalBands], None, True)

def calculateTimeDifference(modisImage, landsatImage):
    modisDate = ee.Date(modisImage.date())
    landsatDate = ee.Date(landsatImage.date())
    return landsatDate.difference(modisDate, 'day').abs()

def findClosestLandsat(modisImage,landsat):
    modisDate = ee.Date(modisImage.date())
    landsatImagesInRange = landsat.filterDate(modisDate.advance(-60, 'day'), modisDate.advance(60, 'day'))
    sortedLandsat = landsatImagesInRange.map(lambda landsatImage: landsatImage.set('time_difference', calculateTimeDifference(modisImage, landsatImage))).sort('time_difference')
    closestLandsatImage = ee.Image(sortedLandsat.first())
    return modisImage.addBands(closestLandsatImage).set('MODIS_Time', modisDate.format('YYYY-MM-dd HH:mm')).set('DATE_ACQUIRED', modisDate.format('YYYY-MM-dd')).set('Landsat_Time', closestLandsatImage.get('Landsat_Time'))

def downscale(date, clip_roi, Modis, MODIS_Ref_250, MODIS_Ref_500, ERA5):
    st.write('Date',date)
    elevation = DEM.clip(clip_roi)
    elevation = LandsatUpscale(elevation).rename('Elevation')
    DOY = pd.to_datetime(date).day_of_year
    DOY_image = ee.Image.constant(DOY).clip(clip_roi).reproject(elevation.projection()).rename('DOY')
    DOY_image = LandsatUpscale(DOY_image)
    
    start = ee.Date(date)
    end = start.advance(10,'day')

    Landsat_Coll = L8.merge(L9).sort('system:time_start').filterDate(start, end).filterBounds(clip_roi)
    landsat = Landsat_Coll.map(cloudMask).map(LandsatUpscale).map(applyScaleFactors).map(NDVI_NDBI_NDWI)
    Modis = Modis.filterDate(start, end)
    MODIS_Ref_250 = MODIS_Ref_250.filterDate(start, end).select(['sur_refl_b01', 'sur_refl_b02'])
    MODIS_Ref_500 = MODIS_Ref_500.filterDate(start, end).select(['sur_refl_b03', 'sur_refl_b04', 'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07'])
    Modis = Modis.combine(MODIS_Ref_250).combine(MODIS_Ref_500)

    Modis = Modis.map(addBandsToModis).map(lambda img: downsampledLST(img, clip_roi)).select(['sur_refl_b03', 'sur_refl_b04', 'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07', 'LST_Day_1km'])
    ERA5 = ERA5.filterDate(start, end).select('surface_solar_radiation_downwards_hourly').filter(ee.Filter.eq('hour', 8)).map(lambda img: downsampledLST(img, clip_roi)).map(lambda image: image.set('system:time_start', ee.Date(image.get('system:time_start')).update(hour=0, minute=0, second=0).millis()))
    filterJoin = ee.Filter.equals(leftField='system:time_start', rightField='system:time_start')
    simpleJoin = ee.Join.inner()
    modis_ERA = ee.ImageCollection(simpleJoin.apply(Modis, ERA5, filterJoin))
    modis_ERA_Coll = modis_ERA.map(lambda feature: ee.Image.cat(feature.get('primary'), feature.get('secondary')))
    modisWithClosestLandsat = modis_ERA_Coll.map(lambda modisImage: findClosestLandsat(modisImage, landsat).addBands([elevation, DOY_image]))
    return modisWithClosestLandsat

def Predictions(modisWithClosestLandsat,m):
    data = modisWithClosestLandsat.first().wx.to_xarray(scale=100, crs='EPSG:4326')
    df = data.to_dataframe()
    df.reset_index(inplace=True)
    df.drop(columns=['spatial_ref'], inplace=True)
    df.rename(columns={'surface_solar_radiation_downwards_hourly': 'SSRDH'}, inplace=True)
    df1 = df[bands]
    df1.dropna(inplace=True)

    X_test = scaler_X.transform(df1[bands])
    y_pred = best_model.predict(X_test)
    df1['ANN_LST'] = scaler_y.inverse_transform(y_pred)

    merged_df = df.merge(df1['ANN_LST'], how='left', left_index=True, right_index=True)
    merged_df.set_index(['y', 'x'], inplace=True)
    merged_df = merged_df.to_xarray()
    data['ANN_LST'] = merged_df['ANN_LST']
    data['ANN_LST'].attrs = {'long_name': 'ANN_LST', 'AREA_OR_POINT': 'Area', 'grid_mapping': 'spatial_ref'}
    
    # Plot multiple images in subplots
    min_ = np.nanpercentile(df1['ANN_LST'], 1)
    max_ = np.nanpercentile(df1['ANN_LST'], 99)


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.5))
    im1 = ax1.imshow(merged_df['LST_Day_1km'], cmap='jet', vmin=min_, vmax=max_)
    im2 = ax2.imshow(merged_df['ST_B10'], cmap='jet', vmin=min_, vmax=max_)
    im3 = ax3.imshow(merged_df['ANN_LST'], cmap='jet', vmin=min_, vmax=max_)

    ax1.set_title('MODIS LST')
    ax2.set_title('Landsat LST')
    ax3.set_title('ANN LST')

    fig.colorbar(im1, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=ax3, orientation='vertical', fraction=0.046, pad=0.04)

    # Convert the plot to an image for displaying in Streamlit
    st.pyplot(fig)
    pass



# # Streamlit app# Function to process user input and run the code
# def user_input_map(lat, lon, radius, date):
#     try:
#         date_str = date.strftime('%Y-%m-%d')
#         # Create a Map object
        # Map = geemap.Map()
        # Map.add_basemap("HYBRID")
        
#         # Create a point feature
#         point = ee.Geometry.Point(lon, lat)
        
#         # Create a buffer around the point
#         clip_roi = point.buffer(radius).bounds()
#         Map.center_object(clip_roi, 9)
        
#         # Add the buffer region to the map
#         Map.addLayer(clip_roi, {'color': 'red'}, 'ROI')
        
#         # Display the map in Streamlit
#         Map.to_streamlit(height=600)
        
#         return clip_roi,date_str, Map
#     except Exception as e:
#         st.error(f"An error occurred: {str(e)}")

def user_input_map(lat, lon, buffer_size, date):
    try:
        date_str = date.strftime('%Y-%m-%d')
        
        # Create a Map object
        m = leafmap.Map(center=[lat, lon], zoom=6)

        # Add a basemap
        m.add_basemap('HYBRID')
        
        # Create a point geometry
        point = ee.Geometry.Point(lon, lat)
        
        # Create a buffer around the point
        clip_roi = point.buffer(buffer_size).bounds()
        
        # Display the map in Streamlit
        m.to_streamlit()

        return clip_roi, date_str,m
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


def main():
    # Inputs in the sidebar
    st.sidebar.title("Input Coordinates and Buffer Distance")
    lat = st.sidebar.number_input("Latitude", value=27.2)
    lon = st.sidebar.number_input("Longitude", value=77.45)
    radius = st.sidebar.number_input("Buffer (m)", value=20000)
    date_input = st.sidebar.date_input("Date",value=pd.Timestamp('2023-01-16'))
    
    # Run the code when the user clicks the button
    if st.sidebar.button("Submit"):
        clip_roi,date_str,m=user_input_map(lat, lon, radius, date_input)
        modisWithClosestLandsat = downscale(date_str, clip_roi, Modis, MODIS_Ref_250, MODIS_Ref_500, ERA5)
        Predictions(modisWithClosestLandsat,m)
        st.sidebar.success("Code execution completed successfully!")

if __name__ == "__main__":
    main()
