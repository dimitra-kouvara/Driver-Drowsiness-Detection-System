import numpy as np

# image = [3,2,4,3,4,4,6,5,6,7]
# bin_number = 12

def histEqualizer(image, bin_number):
    HEIGHT,WIDTH= image.shape[0:2]
    hist, bins = np.histogram(image, bin_number, [0,bin_number])
    # print('histogram', hist)
    
    cdf = hist.cumsum()
    # print('cumulative', cdf)

    cdf_normal = cdf * hist.max()/cdf.max()
    # print('cdf normalised', cdf_normal)

    cdf_mod = np.ma.masked_equal(cdf, 0 ) 
    # print('masked', cdf_mod)

    #formula
    cdf_mod = ( (cdf_mod - cdf_mod.min()) * (bin_number - 1) / (HEIGHT * WIDTH - cdf_mod.min()) )
    # print('new_vals', cdf_mod)

    cdf = np.ma.filled(cdf_mod, 0) 
    # print('with zeros', cdf)

    cdf = cdf.round()
    # print(cdf)

    image2 = cdf[image]
    # print (image, '\nvs\n', image2)

    hist2, bins2 = np.histogram(image2, bin_number, [0, bin_number])
    cdf_new = hist2.cumsum()
    # print('new cdf', cdf_new)
    return image2

# histEqualizer(image, bin_number)