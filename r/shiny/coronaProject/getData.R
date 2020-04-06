library(httr)
library(rvest)

res <- GET('https://datahub.io/core/covid-19/r/2.html')
print(res)

read_html(res) %>% 
  html_nodes(xpath = '/html/body/div[6]/div/div/div[6]/table')
