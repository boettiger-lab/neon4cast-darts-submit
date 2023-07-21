library(tidyverse)
library(arrow)
library(fable)

neon <- s3_bucket("neon4cast-targets/neon",
                  endpoint_override = "data.ecoforecast.org",
                  anonymous = TRUE)


# we select the RH table by using its full name, this also contains air temp
# for aquatics sites
remote_taat <- open_dataset(neon$path("RH_30min-basic-DP1.00098.001"))


neon_temp <- remote_taat |>
  mutate(time = as.Date(startDateTime)) |>
  group_by(siteID, time) |>
  summarise(air_tmp = mean(tempRHMean, na.rm = TRUE)) |>
  rename(site_id = siteID) |>
  rename(datetime = time) |>
  collect()

target <- read_csv("https://data.ecoforecast.org/neon4cast-targets/aquatics/aquatics-targets.csv.gz") |>
  pivot_wider(names_from = variable, values_from = observation) |>
  left_join(neon_temp, by = c("site_id", "datetime")) |>
  write_csv("targets.csv.gz")
