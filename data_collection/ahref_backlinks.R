library("RAhrefs")

# -------------- AUTH ---------------
api_key <- "KEY"
RAhrefs::rah_auth(api_key)

# -------------- CONFIG ---------------
urls <- read.csv("input.csv")[,1,drop=TRUE]
backlink_limit = 10
file_prefix = 'output'

# Edge list
link_df <- data.frame(matrix(ncol = 5, nrow = 0))
colnames(link_df) <- c('domain_from',   'domain_to',  'links', 'unique_pages', 'domain_to_rating')

count = 0

for (url in urls) {
  print(paste(count, ": ", url))
  count = count + 1
  
  # downloading data -------------------
  url_backlinks <- RAhrefs::rah_refdomains(
    target = url,
    mode = "subdomains",
    limit = backlink_limit,
    order_by = "backlinks:desc"
  )
  url_backlinks = subset(url_backlinks, select = -c(first_seen, last_visited))
  colnames(url_backlinks) <- c('domain_from', 'links', 'unique_pages', 'domain_to_rating')
  
  for (i in 1:backlink_limit) {
    # swap to & from
    backlink_df = c(domain_to=url, url_backlinks[i,])
    temp = backlink_df[1]
    backlink_df[1] = backlink_df[2]
    backlink_df[2] = temp
    
    link_df[nrow(link_df) + 1,] = backlink_df
  }

  write.csv(link_df, paste(file_prefix, '_edges.csv'), row.names = FALSE)
}

write.csv(link_df, paste(file_prefix, '_edges.csv'), row.names = FALSE)
