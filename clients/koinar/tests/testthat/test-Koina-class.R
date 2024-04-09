#R
bloat_array <- function(arr, n) {
  # Determine the new dimensions, repeating the first dimension `n` times
  new_dim <- dim(arr)
  new_dim[1] <- new_dim[1] * n
  # Create an expanded array
  expanded_array <- array(rep(arr, n), dim = new_dim)
  return(expanded_array)
}


test_that("check Prosit2019 Fig1", {
  
  ## input
  peptide <- "LKEATIQLDELNQK"
  peptide_n <- nchar(peptide) - 1
  
  ## indices of top 10 highest intensities
  ground_truth <- c(31, 13, 25, 52, 34, 43, 37, 46, 40, 58)
  
  koina_instance <- koinar::Koina$new(
    model_name = "Prosit_2019_intensity",
    server_url = "koina.wilhelmlab.org:443",
    ssl = TRUE
  )
  
  koina_instance_fgcz <- koinar::Koina$new(
    model_name = "Prosit_2019_intensity",
    server_url = "dlomix.fgcz.uzh.ch:443",
    ssl = TRUE
  )
  
  input <- list(
    peptide_sequences = array(c(peptide),
                              dim = c(1, 1)),
    collision_energies = array(c(25), dim = c(1, 1)),
    precursor_charges = array(c(1), dim = c(1, 1))
  )
  
  prediction_results <- koina_instance$predict(input, pred_as_df = FALSE, min_intensity = 0)
  
  ## determine indices of top 10 highest intensities
  (prediction_results$intensities |> order(decreasing = TRUE))[seq(1, 10)] -> idx
  
  testthat::expect_equal(idx, ground_truth)
  
  prediction_results_fgcz <- koina_instance_fgcz$predict(input, pred_as_df = FALSE, min_intensity = 0)
  ## determine indices of top 10 highest intensities
  (prediction_results_fgcz$intensities |> order(decreasing = TRUE))[seq(1, 10)] -> idx_fgcz
  
  testthat::expect_equal(idx_fgcz, ground_truth)
  
  ## check y-fragmention annotation
  testthat::expect_equal(protViz::fragmentIon(peptide)[[1]]$y[seq(1, peptide_n)],
                         prediction_results$mz[seq(1, 174, by = 6)][seq(1, peptide_n)],
                         tolerance = 1.0e-04)
  
  testthat::expect_equal(protViz::fragmentIon(peptide)[[1]]$y[seq(1, peptide_n)],
                         prediction_results_fgcz$mz[seq(1, 174, by = 6)][seq(1, peptide_n)],
                         tolerance = 1.0e-04)
})

test_that("Check error: Server unavailable", {
  expect_error(koinar::Koina$new(
    model_name = "Prosit_2019_intensity",
    server_url = "google.com",
  ), "Server is not ready. Response status code: 404")
})

test_that("Check error: Model unavailable", {
  expect_error(koinar::Koina$new(
    model_name = "not_a_valid_model_name"
  ), "ValueError: The specified model is not available at the server. ")
})

test_that("Check error: Missing input", {
  
  koina_instance <- koinar::Koina$new(
    model_name = "Prosit_2019_intensity",
    server_url = "koina.wilhelmlab.org:443",
    ssl = TRUE
  )
  
  input <- list(
    peptide_sequence = array(c("LKEATIQLDELNQK"),
                              dim = c(1, 1)),
    collision_energies = array(c(25), dim = c(1, 1)),
    precursor_charges = array(c(1), dim = c(1, 1))
  )
  
  expect_error(koina_instance$predict(input), 
               "Missing input\\(s\\): peptide_sequences.")
})

test_that("Check batching", {
  
  koina_instance <- koinar::Koina$new(
    model_name = "Prosit_2019_intensity",
    server_url = "koina.wilhelmlab.org:443",
    ssl = TRUE
  )
  
  input_data <- list(
    peptide_sequences = array(c("LKEATIQLDELNQK"),
                              dim = c(1, 1)),
    collision_energies = array(c(25), dim = c(1, 1)),
    precursor_charges = array(c(1), dim = c(1, 1))
  )
  large_input_data = lapply(input_data, bloat_array, 1234)
  
  predictions = koina_instance$predict(large_input_data, pred_as_df = FALSE)
  
  expect_equal(dim(predictions[["intensities"]]), c(1234,174))
})

test_that("Check dataframe input", {
  
  koina_instance <- koinar::Koina$new(
    model_name = "Prosit_2019_intensity",
    server_url = "koina.wilhelmlab.org:443",
    ssl = TRUE
  )
  
  input_data <- list(
    peptide_sequences = array(c("LKEATIQLDELNQK"),
                              dim = c(1, 1)),
    collision_energies = array(c(25), dim = c(1, 1)),
    precursor_charges = array(c(1), dim = c(1, 1))
  )
  large_input_data = lapply(input_data, bloat_array, 1234)
  
  df_input = data.frame(large_input_data)
  pred_df = koina_instance$predict(df_input, pred_as_df = FALSE)
  pred_arr = koina_instance$predict(large_input_data, pred_as_df = FALSE)
  
  expect_equal(pred_df[["intensities"]], pred_arr[["intensities"]], 1e-5)
})

test_that("Check dataframe output", {
  
  koina_instance <- koinar::Koina$new(
    model_name = "Prosit_2019_intensity",
    server_url = "koina.wilhelmlab.org:443",
    ssl = TRUE
  )
  
  input_data <- list(
    peptide_sequences = array(c("LKEATIQLDELNQK"),
                              dim = c(1, 1)),
    collision_energies = array(c(25), dim = c(1, 1)),
    precursor_charges = array(c(1), dim = c(1, 1))
  )
  large_input_data = lapply(input_data, bloat_array, 1234)
  
  df_input = data.frame(large_input_data)
  predictions = koina_instance$predict(df_input)
  
  expect_equal(nrow(predictions), 19*1234)
})