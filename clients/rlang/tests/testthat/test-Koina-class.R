#R

test_that("check Prosit2019 Fig1", {

  ## input
  peptide <- "LKEATIQLDELNQK"
  peptide_n <- nchar(peptide) - 1
  
  ## indices of top 10 highest intensities
  ground_truth <- c(31, 13, 25, 52, 34, 43, 37, 46, 40, 58)
  
  koina_instance <- koina::Koina$new(
    model_name = "Prosit_2019_intensity",
    server_url = "koina.wilhelmlab.org:443",
    ssl = TRUE
  )

  koina_instance_fgcz <- koina::Koina$new(
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
  
  prediction_results <- koina_instance$predict(input)
  
  ## determine indices of top 10 highest intensities
  (prediction_results$intensities |> order(decreasing = TRUE))[seq(1, 10)] -> idx

  testthat::expect_equal(idx, ground_truth)

  prediction_results_fgcz <- koina_instance_fgcz$predict(input)
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
