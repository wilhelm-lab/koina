#R

test_that("check Prosit2019 Fig1", {
  
  # indices of top 10 highest intensities
  ground_truth <- c(31, 13, 25, 52, 34, 43, 37, 46, 40, 58)
  
  koina_instance <- koinar::Koina$new(
    model_name = "Prosit_2019_intensity",
    server_url = "koina.wilhelmlab.org:443",
    ssl = TRUE
  )
  
  input <- list(
    peptide_sequences = array(c("LKEATIQLDELNQK"),
                              dim = c(1, 1)),
    collision_energies = array(c(25), dim = c(1, 1)),
    precursor_charges = array(c(1), dim = c(1, 1))
  )
  
  prediction_results <- koina_instance$predict(input)
  
  ## determine indices of top 10 highest intensities
  (prediction_results$intensities |> order(decreasing = TRUE))[1:10] -> idx

  testthat::expect_equal(idx, ground_truth)
})