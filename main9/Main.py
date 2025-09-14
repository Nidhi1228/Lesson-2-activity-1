import requests
import time



def get_random_facts(url):
    try:
        response = requests.get(url,timeout=5)
        if response.status_code == 200:
           facts = response.json()
           return facts['text']
        else:
           return f"Error: {response.status_code}"
    except Exception as e:
       return f"Request Failed:{e}"   

 
def main():
  url = "https://uselessfacts.jsph.pl/random.json?language=en"
  
  print("Welcome to Random Facts ")
  print("Press 'q' to Exit" )
  
  while True:
    
    print(get_random_facts(url))
    time.sleep(3)


if __name__ == "__main__":
   main()