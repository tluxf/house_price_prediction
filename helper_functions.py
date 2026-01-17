def haversine(lat1, lon1, lat2, lon2):
    """
    Returns the distance in km between the two latitude and longitude pairs
    """
    import numpy as np
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    r = 6371 # radius of earth in km
    a = np.sin((lat1-lat2)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon1-lon2)/2)**2
    c = 2 * np.arctan2(a**0.5, (1-a)**0.5)
    d = r*c
    #haversine equation
    #a = sin²(Δφ / 2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ / 2)
    #c = 2 ⋅ atan2( √a, √(1−a) )
    #d = R ⋅ c
    #φ is latitude, λ is longitude, R is earth’s radius (mean radius = 6,371 km);
    return d


def closest_postcode(County, District, City, Locality, Street, match_dataframe):
    """
    Takes the location details of the listing and picks the closest location and returns the postcode
    Initially searches for postcodes with street, locality, city, district and county
    If none found, searches again excluding street, then locality, and so on until it searches whole dataframe
    Returns postcode as a string
    Should always return a postcode, but if none found it will return string 'No match' so the code keeps running 
    """
    partial_matches = match_dataframe[(match_dataframe['County']==County) 
                    & (match_dataframe['District']==District) 
                    & (match_dataframe['City']==City) 
                    & (match_dataframe['Locality']==Locality) 
                    & (match_dataframe['Street']==Street)
                    & match_dataframe['Postcode']]
    if partial_matches['Postcode'].size >0:
        return partial_matches['Postcode'].mode()[0]

    partial_matches = match_dataframe[(match_dataframe['County']==County) 
                    & (match_dataframe['District']==District) 
                    & (match_dataframe['City']==City) 
                    & (match_dataframe['Locality']==Locality) 
                    & match_dataframe['Postcode']]
    if partial_matches['Postcode'].size >0:
        return partial_matches['Postcode'].mode()[0]

    partial_matches = match_dataframe[(match_dataframe['County']==County) 
                    & (match_dataframe['District']==District) 
                    & (match_dataframe['City']==City) 
                    & match_dataframe['Postcode']]
    if partial_matches['Postcode'].size >0:
        return partial_matches['Postcode'].mode()[0]

    partial_matches = match_dataframe[(match_dataframe['County']==County) 
                    & (match_dataframe['District']==District) 
                    & match_dataframe['Postcode']]
    if partial_matches['Postcode'].size >0:
        return partial_matches['Postcode'].mode()[0]

    partial_matches = match_dataframe[(match_dataframe['County']==County) 
                    & match_dataframe['Postcode']]
    if partial_matches['Postcode'].size >0:
        return partial_matches['Postcode'].mode()[0]

    partial_matches = match_dataframe['Postcode']
    if partial_matches['Postcode'].size >0:
        return partial_matches['Postcode'].mode()[0]

    return 'No match'


def fill_empty_postcode(fill_df, match_df):
    """
    Finds missing postcodes in fill_df and fills them in with with data from match_df
    Takes a while to complete
    Returns no value. Updates fill_df in place
    Prints progress
    """
    #indexes of entries with missing postcodes
    index_to_fill = fill_df[fill_df['Postcode'].isnull()].index

    #iterates over indexes and finds closest postcode in match df 
    #inserts matched postcode into fill df
    for index in index_to_fill:
        County = fill_df.loc[index, 'County']
        District = fill_df.loc[index, 'District']
        City = fill_df.loc[index, 'City']
        Locality = fill_df.loc[index, 'Locality']
        Street = fill_df.loc[index, 'Street']

        matched_postcode = closest_postcode(County, District, City, Locality, Street, match_df)
        fill_df.loc[index, 'Postcode'] = matched_postcode

        print(f'index: {index}, {matched_postcode}, {sum(fill_df['Postcode'].isna())} remaining')


def closest_lsoa(postcode_to_find, lsoa_table):
    """
    Finds lsoa and msoa codes for postcode that are as similar as possible to postcode_to_find
    Returns tuple with matched lsoa and matched msoa
    """
    postcode_to_find_working = postcode_to_find

    #checks postcode in lsoa table. If no matches shorten and repeat
    while len(postcode_to_find_working) > 0:
        lsoa_matches = lsoa_table[(lsoa_table['pcds'].str.startswith(postcode_to_find_working))
                                    & lsoa_table['lsoa21cd'] & lsoa_table['msoa21cd']]
        
        if lsoa_matches['pcds'].size > 0:
            matched_lsoa = lsoa_matches.head(1).loc[:,'lsoa21cd'].iloc[0]
            matched_msoa = lsoa_matches.head(1).loc[:,'msoa21cd'].iloc[0]
            matched_pcds = lsoa_matches.head(1).loc[:,'pcds'].iloc[0]
            print(f'{postcode_to_find} lsoa/msoa matched {matched_pcds}')
            return (matched_lsoa,matched_msoa)
        else:
            postcode_to_find_working = postcode_to_find_working[:-1]

    #if no matches, return string for debugging
    return ('no match','no match')


def fill_missing_lsoa(df, lsoa_table):
    """
    finds missing lsoa and msoa values in dataframe df and fills with lsoa and msoa for a similar postcode
    Returns no value. Upades df in place.
    Prints progress
    """
    #index of missing lsoa and msoa
    missing_lsoa = df[df['lsoa'].isnull() | df['msoa'].isnull()].index

    #iterate over missing lsoa indexes and fill values
    for index in missing_lsoa:
        postcode = df.loc[index, 'postcode']
        tup = closest_lsoa(postcode, lsoa_table)
        lsoa = tup[0]
        msoa = tup[1]
        df.loc[index, 'lsoa'] = lsoa
        df.loc[index, 'msoa'] = msoa
        print(f'remaining: {df[df['lsoa'].isnull()]['lsoa'].size}')


def closest_ltla(postcode_to_find, ltla_table):
    """
    Finds ltla codes for postcode that are as similar as possible to postcode_to_find in ltla_table
    Returns string with ltla
    """
    postcode_to_find_working = postcode_to_find

    #checks postcode in lsoa table. If no matches shorten and repeat
    while len(postcode_to_find_working) > 0:
        ltla_matches = ltla_table[(ltla_table['pcds'].str.startswith(postcode_to_find_working))
                                    & ltla_table['ltla22cd']]
        
        if ltla_matches['pcds'].size > 0:
            matched_ltla = ltla_matches.head(1).loc[:,'ltla22cd'].iloc[0]
            matched_pcds = ltla_matches.head(1).loc[:,'pcds'].iloc[0]
            print(f'{postcode_to_find} ltla matched {matched_pcds}')
            return matched_ltla
            break
        else:
            postcode_to_find_working = postcode_to_find_working[:-1]

    #if no matches, return string for debugging
    return ('no match','no match')


def fill_missing_ltla(df, ltla_table):
    """
    finds missing ltla values in dataframe df and fills with a ltla for a similar postcode
    Returns no value. Upades df in place.
    Prints progress
    """
    #index of missing ltla
    missing_ltla = df[df['ltla'].isnull()].index
    
    #iterate over missing lsoa indexes and fill values
    for index in missing_ltla:
        postcode = df.loc[index, 'postcode']
        ltla = closest_ltla(postcode, ltla_table)
        df.loc[index, 'ltla'] = ltla
        print(f'remaining: {df[df['ltla'].isnull()]['ltla'].size}')


def fill_missing_lat_lon(fill_table, lookup_table):
    """
    Checks fill_table for missing lat/lon/dfl values and fills them with a close postcode from source_table
    Returns no value. Updates fill_table in place.
    Prints progress
    """
    #get indexes mising latlon data
    missing_latlon = fill_table[fill_table['latitude'].isnull() | fill_table['longitude'].isnull()].index

    #iterate over missing indices
    for index in missing_latlon:
        #for each missing index find the latlon data of a close postcode
        postcode = fill_table.loc[index, 'postcode']
        lat, lon = find_close_postcode_latlon(postcode, lookup_table)

        #insert the close values into the df
        fill_table.loc[index, 'latitude'] = lat
        fill_table.loc[index, 'longitude'] = lon
        #print progress
        print(f'Lat/Lon. remaining: {sum(fill_table['longitude'].isna())}, index: {index}')


def find_close_postcode_latlon(postcode, lookup_table):
    """
    Finds a closely matching postcode in lookup_table and returns it
    """
    postcode_working = postcode[:-1]

    while len(postcode_working) > 0:
        #matched postcodes found
        matched_postcodes = lookup_table[(lookup_table['postcode'].str.startswith(postcode_working) 
                             & lookup_table['latitude'] & lookup_table['longitude'])]
        
        #if matched postcodes, return lat/lon of them
        #if not matches, shorten postcode and repeat
        if matched_postcodes['postcode'].size>0:
            matched_lat = matched_postcodes.head(1).loc[:,'latitude'].iloc[0]
            matched_lon = matched_postcodes.head(1).loc[:,'longitude'].iloc[0]
            return (matched_lat,matched_lon)
        else:
            postcode_working = postcode_working[:-1]
            
    #if no match found, return None     
    return None
   

def london_distance(lat_series, lon_series):
    """
    Uses haversine equation to calculate the distance to london for each listing
    Returns a list of distances
    """
    import numpy as np
    london_lat, london_lon = 51.500632, -0.124422
    distance_from_london = []
    for i in range(lat_series.size):    
        if np.isnan(lat_series[i]) or np.isnan(lon_series[i]):
            distance_from_london.append(None)
        else:
            lat, lon = lat_series[i], lon_series[i]
            distance = haversine(lat,lon,london_lat,london_lon)
            distance_from_london.append(distance)
        print(f'{i}/{lat_series.size}')
    return distance_from_london
    
    
