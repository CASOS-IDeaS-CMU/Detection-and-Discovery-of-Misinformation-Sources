library("RAhrefs")

# -------------- AUTH ---------------
api_key <- "KEY"
RAhrefs::rah_auth(api_key)

# -------------- CONFIG ---------------
urls <- read.csv("input.csv")[,1,drop=TRUE]
outlink_limit = 100
file_prefix = 'output'

# Edge list
link_df <- data.frame(matrix(ncol = 5, nrow = 0))
colnames(link_df) <- c('domain_from',   'domain_to',  'links', 'unique_pages', 'domain_to_rating')

count = 0

for (url in urls) {
  print(paste(count, ": ", url))
  count = count + 1
  
  # downloading data -------------------
  url_outlinks <- try(RAhrefs::rah_linked_domains(
    target = url,
    mode = "subdomains",
    limit = outlink_limit,
    order_by = "links:desc"
  ))
  if(inherits(url_outlinks, "try-error"))
  {
    #error handling code, maybe just skip this iteration using
    print(paste("Failed outlinks: ",url))
    
    next
  }
  
  url_outlinks = subset(url_outlinks, select = -c(domain_to_ahrefs_top))
  for (j in 1:outlink_limit) {
    link_df[nrow(link_df) + 1,] = c(url_outlinks[j,])
  }
  
  write.csv(link_df, paste(file_prefix, '_edges.csv'), row.names = FALSE)
}

write.csv(link_df, paste(file_prefix, '_edges.csv'), row.names = FALSE)
