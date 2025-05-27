import requests
from statistics import mode
import math

def getPolyline(source_lat, source_long, dest_lat, dest_long, API_KEY):
    polylines_arr = []

    params = {
        "origin": f"{source_lat},{source_long}",
        "destination": f"{dest_lat},{dest_long}",
        "transportMode": "car",
        "lang": "en-gb",
        "return": "polyline",
        "alternatives": 3,  # Request three alternate routes
        "apiKey": API_KEY,
    }

    try:
        result = requests.get("https://router.hereapi.com/v8/routes", params=params)
        result.raise_for_status()  # Raise an exception for HTTP errors

        # Extract the polylines from the response
        for obj in result.json().get("routes", []):
            polylines_arr.append(obj["sections"][0]["polyline"])

    except requests.exceptions.RequestException as e:
        print(f"Error fetching polyline data: {e}")
        return []

    return polylines_arr


def getTraffic(polyline, API_KEY, radius = 50):

  params = {
              "locationReferencing": "shape", # Request three alternate routes
              "in": f"corridor:{polyline};r={radius}",
              "apiKey": API_KEY,
            }

  result_json = requests.get("https://data.traffic.hereapi.com/v7/flow", params=params)

  # getting the average
  length = len(result_json.json()["results"])
  sum = 0
  temp = 0
  for i in range(length) :
    temp = result_json.json()["results"][i]["currentFlow"]["jamFactor"]
    if(temp != 0):
      sum += temp

  return math.floor(sum) // length


def getWeather(dest_lat, dest_long, API_KEY):

  params = {
              "product": "observation",
              "latitude": dest_lat,
              "longitude": dest_long,
              "apiKey": API_KEY,
            }

  result_json = requests.get("https://weather.ls.hereapi.com/weather/1.0/report.json", params=params)

  length = len(result_json.json()["observations"]["location"])
  sum = 0

  for i in range(length):
    temp = result_json.json()["observations"]["location"][i]['observation'][0]["visibility"]
    if temp != "*":
      sum += int(float(temp))
      
  return math.floor(sum)//length