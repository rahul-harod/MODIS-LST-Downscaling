import wxee
import base64
import ee
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import pandas as pd
import os
import joblib
import Landsat_S2_data
# import leafmap
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import model_from_json
import streamlit as st
from google.oauth2 import service_account
from ee import oauth
import resnet

def add_logo():
    st.sidebar.image("iitb_logo.png", width=200)

add_logo()

"# MODIS LST Downscaling"
def get_auth():
    service_account_keys=st.secrets['ee_keys']
    credentials=service_account.Credentials.from_service_account_info(service_account_keys,scopes=oauth.SCOPES)
    ee.Initialize(credentials)
    return 'Successfully sync to GEE'
    
get_auth()    


ee.Initialize()
selected_model='ANN'
selected_lst_type = 'Aqua_day'

targetProjection = ee.Projection('EPSG:32643')
ERA5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
DEM = ee.Image("USGS/SRTMGL1_003")
L8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
L9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")

lst_paths = {
    'Terra_day': {
        'Modis': "MODIS/061/MOD11A1",
        'MODIS_Ref_500': "MODIS/061/MOD09GA",
        'LST_band':'LST_Day_1km',
    },
    'Aqua_day': {
        'Modis': "MODIS/061/MYD11A1",
        'MODIS_Ref_500': "MODIS/061/MYD09GA",
        'LST_band':'LST_Day_1km',
    },
    'Terra_night': {
        'Modis': "MODIS/061/MOD11A1",
        'MODIS_Ref_500': "MODIS/061/MOD09GA",
        'LST_band':'LST_Night_1km'
    },
    'Aqua_night': {
        'Modis': "MODIS/061/MYD11A1",
        'MODIS_Ref_500': "MODIS/061/MYD09GA",
        'LST_band':'LST_Night_1km'
    },
}

def Upscale(img):
    return img.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024).reproject(crs=targetProjection, scale=100)

def addBandsToModis(img,LST_band):
    thermalBands = img.select(LST_band).multiply(0.02).rename('MODIS_LST')
    opticalBands = img.select('sur_refl_b.*').multiply(0.0001)
    return img.addBands([thermalBands, opticalBands], None, True)

def downsampledMODIS_LST(img,clip_roi):
    original_lst=img.select('MODIS_LST').rename('Original_MODIS_LST')
    return img.resample('bilinear').reproject(crs=targetProjection, scale=100).addBands(original_lst).clip(clip_roi)
    
def calculateTimeDifference(modisImage, landsatImage):
    modisDate = modisImage.date()
    landsatDate = landsatImage.date()
    return landsatDate.difference(modisDate, 'day').abs()

def add_time_difference(modisImage,landsat_image):
    difference = calculateTimeDifference(modisImage, landsat_image)
    inverted_time_diff = ee.Image.constant(1).divide(difference)
    return landsat_image.addBands(inverted_time_diff.rename('time_difference')).float()

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

def downscale(date,point, clip_roi, Modis, MODIS_Ref_500,LST_band):
    st.write('Date',date)
    elevation = DEM.clip(clip_roi)
    elevation = Upscale(elevation).rename('Elevation')
    
    start = ee.Date(date)
    end = start.advance(1,'day')

    landsat = Landsat_S2_data.Harmonized_LS(start,clip_roi)
    Modis = Modis.filterDate(start, end).select(LST_band)
    MODIS_Ref_500 = MODIS_Ref_500.filterDate(start, end).select(['sur_refl_b07'])
    Modis = Modis.combine(MODIS_Ref_500).first()
    
    # Map the time difference calculation function over the Landsat collection
    sorted_landsat = landsat.map(lambda landsat_image: landsat_image.addBands(add_time_difference(Modis, landsat_image)))

    # Identify the closest Landsat image based on the inverted time difference
    closest_landsat_image = sorted_landsat.qualityMosaic('time_difference').clip(clip_roi)

    Modis = addBandsToModis(Modis,LST_band)
    Modis=downsampledMODIS_LST(Modis, clip_roi).select([ 'sur_refl_b07','MODIS_LST','Original_MODIS_LST'])
    modisWithClosestLandsat = Modis.addBands([elevation, closest_landsat_image])
    return modisWithClosestLandsat

scaler_X = None
scaler_y = None
best_model = None

def load_model_and_scaler_ANN(model_name,selected_lst_type):
    global scaler_X, scaler_y, best_model
    model_dir = f"Models/{model_name}/"
    scaler_X = joblib.load(model_dir + "152_ANN_scaler_X_"+selected_lst_type+".pkl")
    scaler_y = joblib.load(model_dir + "152_ANN_scaler_y_"+selected_lst_type+".pkl")
    
    with open(model_dir + "152_ANN_architecture_night.json", "r") as json_file:
        loaded_model_json = json_file.read()

    best_model = model_from_json(loaded_model_json)
    best_model.load_weights(model_dir + "152_ANN_Weights_"+selected_lst_type+".h5")

def Predictions_ANN(modisWithClosestLandsat,date_str,selected_lst_type,selected_model):
    bands_ANN=['MODIS_LST', 'sur_refl_b07', 'NDVI', 'NDBI', 'NDWI', 'Elevation']

    data = modisWithClosestLandsat.wx.to_xarray(scale=100, crs='EPSG:4326')
    df = data.to_dataframe()
    df.reset_index(inplace=True)
    df.drop(columns=['spatial_ref'], inplace=True)
    df1 = df[bands_ANN]
    df1.dropna(inplace=True)
    
    load_model_and_scaler_ANN(selected_model,selected_lst_type)
    X_test = scaler_X.transform(df1[bands_ANN])
    y_pred = best_model.predict(X_test)
    df1['ANN_LST'] = scaler_y.inverse_transform(y_pred)

    merged_df = df.merge(df1['ANN_LST'], how='left', left_index=True, right_index=True)
    merged_df.set_index(['y', 'x'], inplace=True)
    merged_df = merged_df.to_xarray()
    data['ANN_LST'] = merged_df['ANN_LST']
    data['ANN_LST'].attrs = {'long_name': 'ANN LST (K)', 'AREA_OR_POINT': 'Area', 'grid_mapping': 'spatial_ref'}
    data['Original_MODIS_LST'].attrs = {'long_name': 'MODIS LST (K)', 'AREA_OR_POINT': 'Area', 'grid_mapping': 'spatial_ref'}
    
    # Plot multiple images in subplots
    min_1 = np.nanpercentile(df1['ANN_LST'], 1)
    max_1 = np.nanpercentile(df1['ANN_LST'], 99)

    min_2 = np.nanpercentile(df1['MODIS_LST'], 1)
    max_2 = np.nanpercentile(df1['MODIS_LST'], 99)

    min_ = min(min_1, min_2)
    max_ = max(max_1, max_2)
    
    fig, (ax1, ax2,cax) = plt.subplots(ncols=3 ,figsize=(8, 3.5),gridspec_kw={"width_ratios":[1,1,0.05]})
    # fig.subplots_adjust(wspace=0.1)
    im1 = data['Original_MODIS_LST'].plot(ax=ax1, cmap='jet', vmin=min_, vmax=max_,add_colorbar=False)
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
    st.markdown(get_nc_download_link(data[['Original_MODIS_LST','ANN_LST']],file_name=selected_lst_type+'_Downscaled_LST_'+date_str+'_'+selected_model+'.nc'), unsafe_allow_html=True)
    st.markdown(get_png_download_link(fig, file_name=selected_lst_type+'_Downscaled_LST_Map_'+date_str+'_'+selected_model+'.png'), unsafe_allow_html=True)
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

        return point,clip_roi, date_str
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


def main():
    global selected_lst_type,Modis, MODIS_Ref_250, MODIS_Ref_500, ERA5,ERA_hour,LST_band,selected_model
    # Inputs in the sidebar
    st.sidebar.title("Enter Search Criteria")
    lat = st.sidebar.number_input("Latitude", value=27.2)
    lon = st.sidebar.number_input("Longitude", value=77.45)
    radius = st.sidebar.number_input("Square Buffer distance (m)", value=20000)
    date_input = st.sidebar.date_input("Date", value=pd.Timestamp('2023-01-16'))
    

    lst_types = ['Aqua_day', 'Aqua_night', 'Terra_day', 'Terra_night']
    selected_lst_type = st.sidebar.selectbox("Select LST Type", lst_types, index=lst_types.index(selected_lst_type))

    Model_types = ['ANN' ,'ResNet']
    selected_model = st.sidebar.selectbox("Select Model", Model_types, index=Model_types.index(selected_model))

    # Update variables based on the selected LST type
    Modis = ee.ImageCollection(lst_paths[selected_lst_type]['Modis'])
    MODIS_Ref_500 = ee.ImageCollection(lst_paths[selected_lst_type]['MODIS_Ref_500'])
    LST_band=lst_paths[selected_lst_type]['LST_band']
    st.write(selected_lst_type+': '+selected_model)
    # Run the code when the user clicks the button
    if st.sidebar.button("Submit"):
        point,clip_roi,date_str=user_input_map(lat, lon, radius, date_input)
        modisWithClosestLandsat = downscale(date_str,point, clip_roi, Modis, MODIS_Ref_500,LST_band)
        if selected_model in ['ANN']:
            Predictions_ANN(modisWithClosestLandsat,date_str,selected_lst_type,selected_model)
            
        # if selected_model in ['ResNet_SMWA', 'ResNet_L2']:
        #     Predictions_ResNet(modisWithClosestLandsat,date_str,selected_lst_type,selected_model)
            
        st.sidebar.success("Code execution completed successfully!")

if __name__ == "__main__":
    main()
