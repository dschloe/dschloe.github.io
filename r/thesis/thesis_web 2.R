library(httr) 
library(urltools) 
library(rvest) 
library(tidyverse) 
library(XML)
library(writexl)

#list-container > a:nth-child(1) > div > div


res <- GET(url = 'https://www.techshake.asia/startups', 
           query = list(page=1))

res %>% 
  read_html() %>% 
  html_node(css = "div.startup-items") %>% 
  html_nodes("a") %>% 
  html_attr("href") %>% unique()


name_of_company <- res %>% 
  read_html() %>% 
  html_nodes(xpath = "//*[@id='list-container']/a[11]/div/div/div[2]/div/div[1]") %>% 
  html_text()

email_address <- res %>% 
  read_html() %>%
  html_nodes(xpath = "//*[@id='list-container']/a[11]/div/div/div[3]/dl/dd[2]/text()") %>% 
  html_text()

email_address



website <- res %>% 
  read_html() %>% 
  html_nodes(xpath = "//*[@id='list-container']/a[11]/div/div/div[3]/dl/dd[3]/text()") %>% 
  html_text()

getStartupDf <- function(url, page) {
  # check
  res <- GET(url = url, query = list(page=page))
  
  result <- data.frame()
  for (i in 1:30) {
    name_of_company <- res %>% 
      read_html() %>% 
      html_nodes(xpath = paste0("//*[@id='list-container']/a[",i,"]/div/div/div[2]/div/div[1]")) %>% 
      html_text()
    
    email_address <- res %>% 
      read_html() %>%
      html_nodes(xpath = paste0("//*[@id='list-container']/a[", i, "]/div/div/div[3]/dl/dd[2]/text()")) %>% 
      html_text()
    
    if (rlang::is_empty(email_address)) {
      cat("name of company is ", name_of_company)
      email_address <- "none"
    }

    website <- res %>% 
      read_html() %>% 
      html_nodes(xpath = paste0("//*[@id='list-container']/a[", i, "]/div/div/div[3]/dl/dd[3]/text()")) %>% 
      html_text()
    
    if (rlang::is_empty(website)) {
      cat("name of company is ", name_of_company)
      website <- "none"
    }
    
    cat("the page num is ", page, "div num is: ", i, "\n")
    df <- data.frame(name_of_company = name_of_company, 
                     email_address = email_address, 
                     website = website)
    result <- rbind(result, df)
  }
  return(result)
}



url <- 'https://www.techshake.asia/startups'

result <- data.frame()
for (i in 1:30) {
  df <- getStartupDf(url, page=i)
  result <- rbind(result, df)
}

write_xlsx(result, "startup_list.xlsx")




