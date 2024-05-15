import wxee
import base64
import ee
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import pandas as pd
import os
import joblib
# import leafmap
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import model_from_json
import streamlit as st
from google.oauth2 import service_account
from ee import oauth

"# MODIS LST Downscaling"
def get_auth():
    service_account_keys=st.secrets['ee_keys']
    credentials=service_account.Credentials.from_service_account_info(service_account_keys,scopes=oauth.SCOPES)
    ee.Initialize(credentials)
    return 'Successfully sync to GEE'
    
get_auth()    

# Initialize global variables
targetProjection = ee.Projection('EPSG:32643')
ERA5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
DEM = ee.Image("USGS/SRTMGL1_003")
L8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
L9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")

lst_paths = {
    'Aqua_daytime': {
        'Modis': "MODIS/061/MYD11A1",
        'MODIS_Ref_250': "MODIS/061/MYD09GQ",
        'MODIS_Ref_500': "MODIS/061/MYD09GA",
        'ERA_hour':8,
        'LST_band':'LST_Day_1km',
    },
    'Aqua_nighttime': {
        'Modis': "MODIS/061/MYD11A1",
        'MODIS_Ref_250': "MODIS/061/MYD09GQ",
        'MODIS_Ref_500': "MODIS/061/MYD09GA",
        'ERA_hour':20,
        'LST_band':'LST_Night_1km'
    },
    'Terra_daytime': {
        'Modis': "MODIS/061/MOD11A1",
        'MODIS_Ref_250': "MODIS/061/MOD09GQ",
        'MODIS_Ref_500': "MODIS/061/MOD09GA",
        'ERA_hour':5,
        'LST_band':'LST_Day_1km',
    },
    'Terra_nighttime': {
        'Modis': "MODIS/061/MOD11A1",
        'MODIS_Ref_250': "MODIS/061/MOD09GQ",
        'MODIS_Ref_500': "MODIS/061/MOD09GA",
        'ERA_hour':17,
        'LST_band':'LST_Night_1km'
    }
}

# Initialize variables with default paths
selected_lst_type = 'Aqua_daytime'
ERA_hour=8
LST_band='LST_Day_1km'
Modis = ee.ImageCollection(lst_paths[selected_lst_type]['Modis'])
MODIS_Ref_250 = ee.ImageCollection(lst_paths[selected_lst_type]['MODIS_Ref_250'])
MODIS_Ref_500 = ee.ImageCollection(lst_paths[selected_lst_type]['MODIS_Ref_500'])


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

def downsampledLST(img):
    return img.resample('bilinear').reproject(crs=targetProjection, scale=100)
    
def downsampledMODIS_LST(img):
    original_lst=img.select('LST_Day_1km').rename('Original_MOD_LST')
    return img.resample('bilinear').reproject(crs=targetProjection, scale=100).addBands(original_lst)
    
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

def addBandsToModis(img,LST_band):
    thermalBands = img.select(LST_band).multiply(0.02).rename('LST_Day_1km')
    opticalBands = img.select('sur_refl_b.*').multiply(0.0001)
    return img.addBands([thermalBands, opticalBands], None, True)

def calculateTimeDifference(modisImage, landsatImage):
    modisDate = ee.Date(modisImage.date())
    landsatDate = ee.Date(landsatImage.date())
    return landsatDate.difference(modisDate, 'day').abs()

def findClosestLandsat(modisImage,landsat):
    modisDate = ee.Date(modisImage.date())
    landsatImagesInRange = landsat.filterDate(modisDate.advance(-17, 'day'), modisDate.advance(17, 'day'))
    sortedLandsat = landsatImagesInRange.map(lambda landsatImage: landsatImage.set('time_difference', calculateTimeDifference(modisImage, landsatImage))).sort('time_difference')
    closestLandsatImage = ee.Image(sortedLandsat.first())
    return modisImage.addBands(closestLandsatImage).set('MODIS_Time', modisDate.format('YYYY-MM-dd HH:mm')).set('DATE_ACQUIRED', modisDate.format('YYYY-MM-dd')).set('Landsat_Time', closestLandsatImage.get('Landsat_Time'))

def downscale(date, clip_roi, Modis, MODIS_Ref_250, MODIS_Ref_500, ERA5,ERA_hour,LST_band):
    st.write('Date',date)
    elevation = DEM.clip(clip_roi)
    elevation = LandsatUpscale(elevation).rename('Elevation')
    DOY = pd.to_datetime(date).day_of_year
    DOY_image = ee.Image.constant(DOY).clip(clip_roi).reproject(elevation.projection()).rename('DOY')
    DOY_image = LandsatUpscale(DOY_image)
    
    start = ee.Date(date)
    end = start.advance(1,'day')

    Landsat_Coll = L8.merge(L9).sort('system:time_start').filterDate(start.advance(-32,'days'), end.advance(32,'days')).filterBounds(clip_roi)
    landsat = Landsat_Coll.map(cloudMask).map(LandsatUpscale).map(applyScaleFactors).map(NDVI_NDBI_NDWI)
    Modis = Modis.filterDate(start, end)
    MODIS_Ref_250 = MODIS_Ref_250.filterDate(start, end).select(['sur_refl_b01', 'sur_refl_b02'])
    MODIS_Ref_500 = MODIS_Ref_500.filterDate(start, end).select(['sur_refl_b03', 'sur_refl_b04', 'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07'])
    Modis = Modis.combine(MODIS_Ref_250).combine(MODIS_Ref_500)

    Modis = Modis.map(lambda img: addBandsToModis(img,LST_band).map(lambda img: downsampledMODIS_LST(img, clip_roi)).select(['sur_refl_b03', 'sur_refl_b04', 'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07', 'LST_Day_1km','Original_MOD_LST'])
    ERA5 = ERA5.filterDate(start, end).select('surface_solar_radiation_downwards_hourly').filter(ee.Filter.eq('hour', ERA_hour)).map(lambda img: downsampledLST(img, clip_roi)).map(lambda image: image.set('system:time_start', ee.Date(image.get('system:time_start')).update(hour=0, minute=0, second=0).millis()))
    filterJoin = ee.Filter.equals(leftField='system:time_start', rightField='system:time_start')
    simpleJoin = ee.Join.inner()
    modis_ERA = ee.ImageCollection(simpleJoin.apply(Modis, ERA5, filterJoin))
    modis_ERA_Coll = modis_ERA.map(lambda feature: ee.Image.cat(feature.get('primary'), feature.get('secondary')))
    modisWithClosestLandsat = modis_ERA_Coll.map(lambda modisImage: findClosestLandsat(modisImage, landsat).addBands([elevation, DOY_image]))
    return modisWithClosestLandsat


def get_nc_download_link(ds, file_name='Downscaled_LST.nc'):
    nc_bytes = ds.to_netcdf()  # Convert xarray dataset to NetCDF bytes
    nc_b64 = base64.b64encode(nc_bytes).decode()  # Encode NetCDF bytes to base64
    href = f'<a href="data:file/nc;base64,{nc_b64}" download="{file_name}">Download NetCDF file</a>'
    return href

def get_png_download_link(f, file_name='Downscaled_LST_Map.png'):
    # Save the plot as PNG
    f.savefig(file_name, dpi=600)
    
    # Read the saved PNG file
    with open(file_name, 'rb') as f_png:
        png_bytes = f_png.read()

    # Encode PNG bytes to base64
    png_b64 = base64.b64encode(png_bytes).decode()
    
    # Generate the download link
    href = f'<a href="data:file/png;base64,{png_b64}" download="{file_name}">Download PNG file</a>'
    return href

    
def Predictions(modisWithClosestLandsat,date_str,selected_lst_type):
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
    data['ANN_LST'].attrs = {'long_name': 'LST (K)', 'AREA_OR_POINT': 'Area', 'grid_mapping': 'spatial_ref'}
    data['Original_MOD_LST'].attrs = {'long_name': 'LST (K)', 'AREA_OR_POINT': 'Area', 'grid_mapping': 'spatial_ref'}
    
    # Plot multiple images in subplots
    min_ = np.nanpercentile(df1['ANN_LST'], 1)
    max_ = np.nanpercentile(df1['ANN_LST'], 99)
    
    fig, (ax1, ax2,cax) = plt.subplots(ncols=3 ,figsize=(8, 3.5),gridspec_kw={"width_ratios":[1,1,0.05]})
    # fig.subplots_adjust(wspace=0.1)
    im1 = data['Original_MOD_LST'].plot(ax=ax1, cmap='jet', vmin=min_, vmax=max_,add_colorbar=False)
    im2 = data['ANN_LST'].plot(ax=ax2, cmap='jet', vmin=min_, vmax=max_,add_colorbar=False)
    
    ax1.set_title('MODIS LST')
    ax2.set_title('ANN LST')
    ip = InsetPosition(ax2, [1.05,0,0.05,1]) 
    cax.set_axes_locator(ip)
    cbar=fig.colorbar(im2, cax=cax, ax=[ax1,ax2])
    cbar.set_label('LST in Kelvin', size=12)

    for ax in (ax1, ax2):
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.tight_layout()
    
    # Convert the plot to an image for displaying in Streamlit
    st.pyplot(fig)
    st.markdown(get_nc_download_link(data[['LST_Day_1km','ANN_LST']],file_name=selected_lst_type+'_Downscaled_LST_'+date_str+'.nc'), unsafe_allow_html=True)
    st.markdown(get_png_download_link(fig, file_name=selected_lst_type+'_Downscaled_LST_Map_'+date_str+'.png'), unsafe_allow_html=True)
    pass

def user_input_map(lat, lon, buffer_size, date):
    try:
        date_str = date.strftime('%Y-%m-%d')
        
        # Create a Map object
        # m = leafmap.Map(center=[lat, lon], zoom=6)

        # # Add a basemap
        # m.add_basemap('HYBRID')
        
        # Create a point geometry
        point = ee.Geometry.Point(lon, lat)
        
        # Create a buffer around the point
        clip_roi = point.buffer(buffer_size).bounds()
        
        # Display the map in Streamlit
        # m.to_streamlit()

        return clip_roi, date_str
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


def main():
    global selected_lst_type, Modis, MODIS_Ref_250, MODIS_Ref_500,ERA_hour,LST_band
    
    # Inputs in the sidebar
    st.sidebar.title("Enter Search Criteria")
    lat = st.sidebar.number_input("Latitude", value=27.2)
    lon = st.sidebar.number_input("Longitude", value=77.45)
    radius = st.sidebar.number_input("Square Buffer distance (m)", value=20000)
    date_input = st.sidebar.date_input("Date", value=pd.Timestamp('2023-01-16'))
    
    lst_types = ['Aqua_daytime', 'Aqua_nighttime', 'Terra_daytime', 'Terra_nighttime']
    selected_lst_type = st.sidebar.selectbox("Select LST Type", lst_types, index=lst_types.index(selected_lst_type))
    
    # Update variables based on the selected LST type
    Modis = ee.ImageCollection(lst_paths[selected_lst_type]['Modis'])
    MODIS_Ref_250 = ee.ImageCollection(lst_paths[selected_lst_type]['MODIS_Ref_250'])
    MODIS_Ref_500 = ee.ImageCollection(lst_paths[selected_lst_type]['MODIS_Ref_500'])
    ERA_hour=lst_paths[selected_lst_type]['ERA_hour']
    LST_band=lst_paths[selected_lst_type]['LST_band']
    # st.write("Path",lst_paths[selected_lst_type]['Modis'])
    # Run the code when the user clicks the button
    if st.sidebar.button("Submit"):
        clip_roi,date_str=user_input_map(lat, lon, radius, date_input)
        modisWithClosestLandsat = downscale(date_str, clip_roi, Modis, MODIS_Ref_250, MODIS_Ref_500, ERA5,ERA_hour,LST_band)
        Predictions(modisWithClosestLandsat,date_str,selected_lst_type)
        st.sidebar.success("Code execution completed successfully!")

if __name__ == "__main__":
    main()
