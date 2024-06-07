import ee

# Define the correction factors
correctionFactors = {
    'B2': {'slopeA': 0.9778, 'offsetA': -0.004, 'slopeB': 0.9778, 'offsetB': -0.004},
    'B3': {'slopeA': 1.0053, 'offsetA': -0.0009, 'slopeB': 1.0075, 'offsetB': -0.0008},
    'B4': {'slopeA': 0.9765, 'offsetA': 0.0009, 'slopeB': 0.9761, 'offsetB': 0.001},
    'B8A': {'slopeA': 0.9983, 'offsetA': -0.0001, 'slopeB': 0.9966, 'offsetB': 0.000},
    'B11': {'slopeA': 0.9987, 'offsetA': -0.0011, 'slopeB': 1.000, 'offsetB': -0.0003},
    'B12': {'slopeA': 1.003, 'offsetA': -0.0012, 'slopeB': 0.9867, 'offsetB': 0.0004}
}

def applyCorrectionS2B(image):
    bandNames = ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12']
    correctedBands = [image.select([bandName])
                      .multiply(0.0001)
                      .multiply(correctionFactors[bandName]['slopeB'])
                      .add(correctionFactors[bandName]['offsetB'])
                      .rename(bandName) for bandName in bandNames]
    correctedImage = ee.Image.cat(correctedBands).copyProperties(image, image.propertyNames())
    return correctedImage

def applyCorrectionS2A(image):
    bandNames = ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12']
    correctedBands = [image.select([bandName])
                      .multiply(0.0001)
                      .multiply(correctionFactors[bandName]['slopeA'])
                      .add(correctionFactors[bandName]['offsetA'])
                      .rename(bandName) for bandName in bandNames]
    correctedImage = ee.Image.cat(correctedBands).copyProperties(image, image.propertyNames())
    return correctedImage

def CloudMaskS2(img):
    mask = img.select('SCL').lt(8).And(img.neq(1))  # Mask cloud and snow/Ice
    return img.updateMask(mask)

def cloudMask(img):
    qaBand = img.select('Fmask').int()
    cloudBitMask = 1 << 1  # Bit 3 represents cloud
    cloudShadowBitMask = 1 << 3  # Bit 4 represents cloud shadow
    cloudMask = qaBand.bitwiseAnd(cloudBitMask).neq(0)
    cloudShadowMask = qaBand.bitwiseAnd(cloudShadowBitMask).neq(0)
    combinedMask = cloudMask.Or(cloudShadowMask)
    return img.updateMask(combinedMask.Not())


def Harmonized_LS(date_str,roi):

    SDate = date_str.advance(-32, 'day')
    EDate = date_str.advance(32, 'day')

    sentinel2B = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(SDate, EDate).filterBounds(roi)\
                    .filter(ee.Filter.eq('SPACECRAFT_NAME', 'Sentinel-2B')).map(CloudMaskS2)
    sentinel2A = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(SDate, EDate).filterBounds(roi)\
                    .filter(ee.Filter.eq('SPACECRAFT_NAME', 'Sentinel-2A')).map(CloudMaskS2)

    correctedS2A = sentinel2A.select(['B2', 'B3', 'B4', 'B8A', 'B11', 'B12']).map(applyCorrectionS2A)
    correctedS2B = sentinel2B.select(['B2', 'B3', 'B4', 'B8A', 'B11', 'B12']).map(applyCorrectionS2B)

    corrected_S2 = correctedS2A.merge(correctedS2B).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)).map(lambda img: img.addBands(img.normalizedDifference(['B8A', 'B4']).rename('NDVI'))
            .addBands(img.normalizedDifference(['B3', 'B8A']).rename('NDWI'))
            .addBands(img.normalizedDifference(['B11', 'B8A']).rename('NDBI'))
            .copyProperties(img, img.propertyNames()))

    HLSL30 = ee.ImageCollection("NASA/HLS/HLSL30/v002").filterDate(SDate, EDate).filterBounds(roi).filter(ee.Filter.lt('CLOUD_COVERAGE', 30)).map(cloudMask).map(lambda img:img.addBands(img.normalizedDifference(['B5', 'B4']).rename('NDVI'))
            .addBands(img.normalizedDifference(['B3', 'B5']).rename('NDWI'))
            .addBands(img.normalizedDifference(['B6', 'B5']).rename('NDBI'))
            .copyProperties(img, img.propertyNames()))

    combined = corrected_S2.merge(HLSL30).sort('system:time_start').select(['NDVI', 'NDWI', 'NDBI'])
    return combined
