# KoinaR

## Contribute
### dev dependencies
```
install.packages(c("roxygen2", "BiocManager", "httr", "jsonlite", "rmarkdown", "testthat", "pdflatex"))
BiocManager::install(c('BiocStyle', 'BiocCheck'))
```
Build vignette with Knit

### Documentation
Use roxygen2 to render the documentation 
`roxygen2::roxygenise()`.

### Tests
We use testthat to run tests. In the `build` tab in Rstudio click on `Test`. 
Make sure you installed the package beforehand

### Verify CodeStyle
Verify code style according to bioconductor guidelines.
`BiocCheck::BiocCheck()`



R CMD build
R CMD check

TODO Adjust output