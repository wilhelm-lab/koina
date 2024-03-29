library(httr)
library(jsonlite)

Koina <- setRefClass(
  "Koina",
  fields = list(
    model_inputs = "list",
    model_outputs = "list",
    batch_size = "numeric",
    response_dict = "list",
    model_name = "character",
    url = "character",
    ssl = "logical",
    disable_progress_bar = "logical",
    client = "ANY", # Placeholder for client object, adjust as needed
    type_convert = "list"
  ),
  methods = list(
    initialize = function(model_name, server_url = "koina.wilhelmlab.org:443", ssl = TRUE, targets = NULL, disable_progress_bar = FALSE) {
      .self$model_inputs <<- list()
      .self$model_outputs <<- list()
      .self$response_dict <<- list()

      .self$model_name <<- model_name
      .self$url <<- server_url
      .self$ssl <<- ssl
      .self$disable_progress_bar <<- disable_progress_bar
      .self$client <<- NA # Initialize your InferenceServerClient here, adjust as needed

      .self$type_convert <<- list(
        FP32 = "float32",
        BYTES = "character",
        INT16 = "integer",
        INT32 = "integer",
        INT64 = "integer"
      )

      # Placeholder for server and model check methods
      .self$is_server_ready()
      .self$is_model_ready()
      
      # Placeholder for methods to get inputs, outputs, and batch size
      .self$get_inputs()
      .self$get_outputs()
      .self$get_batchsize()
    },
    is_server_ready = function() {
      protocol <- ifelse(.self$ssl, "https", "http")
      endpoint <- paste0(protocol, "://", .self$url, "/v2/health/ready")
      response <- httr::GET(endpoint)
      if(httr::status_code(response) == 200) {
        return(TRUE)
      } else {
        stop("Server is not ready. Response status code: ", httr::status_code(response))
      }
    },
    is_model_ready = function() {
      protocol <- ifelse(.self$ssl, "https", "http")
      endpoint <- paste0(protocol, "://", .self$url, "/v2/models/", .self$model_name, "/ready")
      response <- httr::GET(endpoint)
      
      # If the status code is 200, the model is ready.
      if(httr::status_code(response) == 200) {
        return(TRUE)
      } else {
        # It's a good practice to provide specifics about the failure for better debugging.
        content <- httr::content(response, "text", encoding = "UTF-8")
        if(httr::status_code(response) == 400) {
          stop("ValueError: The specified model is not available at the server. ", content)
        } else {
          stop("InferenceServerException: An exception occurred while querying the server for available models. ", content)
        }
      }
    },
    get_inputs = function() {
      protocol <- ifelse(.self$ssl, "https", "http")
      endpoint <- paste0(protocol, "://", .self$url, "/v2/models/", .self$model_name)
      
      # Perform the GET request
      response <- httr::GET(endpoint)
      
      # Check if the request was successful
      if (httr::status_code(response) != 200) {
        stop("InferenceServerException: An exception occurred while querying the server for model inputs.")
      }
      
      # Parse the response
      content <- httr::content(response, "parsed")
      
      # Retrieve inputs from the model's metadata and store them
      if (!is.null(content$inputs)) {
        .self$model_inputs <<- setNames(
          lapply(content$inputs, function(i) list(shape = i$shape, datatype = i$datatype)),
          sapply(content$inputs, function(i) i$name)
        )
      } else {
        stop("InferenceServerException: Unable to retrieve model inputs from the server response.")
      }
    },
    get_outputs = function() {
      protocol <- ifelse(.self$ssl, "https", "http")
      endpoint <- paste0(protocol, "://", .self$url, "/v2/models/", .self$model_name)
      
      # Perform the GET request
      response <- httr::GET(endpoint)
      
      # Check if the request was successful
      if (httr::status_code(response) != 200) {
        stop("InferenceServerException: An exception occurred while querying the server for model metadata.")
      }
      
      # Parse the response
      content <- httr::content(response, "parsed")
      
      # Extract and store outputs from the model's metadata
      if (!is.null(content$outputs)) {
        .self$model_outputs <<- setNames(
          lapply(content$outputs, function(out) list(datatype = out$datatype)),
          sapply(content$outputs, function(out) out$name)
        )
      } else {
        stop("InferenceServerException: Unable to retrieve model outputs from the server response.")
      }
    },
    get_batchsize = function() {
      protocol <- ifelse(.self$ssl, "https", "http")
      endpoint <- paste0(protocol, "://", .self$url, "/v2/models/", .self$model_name, "/config")
      
      response <- httr::GET(endpoint)
      
      if(httr::status_code(response) != 200) {
        stop("InferenceServerException: An exception occurred while querying the server for the max batch size.")
      }
      
      content <- httr::content(response, "parsed")
      
      if (!is.null(content$max_batch_size)) {
        .self$batch_size <<- content$max_batch_size
      } else {
        stop("InferenceServerException: Unable to retrieve max batch size from the server response.")
      }
    },
    predict_batch = function(input_data) {
      # Check if all required inputs are provided
      required_inputs <- names(.self$model_inputs)
      missing_inputs <- setdiff(required_inputs, names(input_data))
      if (length(missing_inputs) > 0) {
        stop("Missing input(s): ", paste(missing_inputs, collapse=", "), ".")
      }
      
      # Construct the endpoint
      protocol <- ifelse(.self$ssl, "https", "http")
      endpoint <- paste0(protocol, "://", .self$url, "/v2/models/", .self$model_name, "/infer")
      
      # Prepare the inputs
      inputs <- lapply(names(input_data), function(name) {
        data_list <- input_data[[name]]
        list(
          name = name,
          shape = dim(data_list),#c(length(data_list), 1), # Assuming the shape from the data list length. Adjust per actual requirements.
          datatype = .self$model_inputs[[name]]$datatype, # Retrieve datatype from the model inputs information
          data = data_list
        )
      })
      
      # Prepare the POST data
      post_data <- list(
        id = Sys.time(), # Example ID generation based on system time. Adjust as needed.
        inputs = inputs
      )
      
      # Convert the list to JSON
      json_data <- jsonlite::toJSON(post_data, auto_unbox = TRUE)
      # Perform the POST request
      response <- httr::POST(
        url = endpoint,
        body = json_data,
        encode = "json",
        httr::add_headers(`Content-Type` = "application/json")
      )
      
      # Check response and return
      if(httr::status_code(response) != 200) {
        stop("Request failed. Status code: ", httr::content(response))
      } else {
        response_json <- httr::content(response, "text", encoding = "UTF-8") # Assuming response is obtained from httr::POST
        return (.self$convert_response_to_list(response_json)) # Use the method to process and return data
      }
    },
    convert_response_to_list = function(json_response) {
      # Ensure that parsed data isn't being simplified to a vector automatically
      parsed_response <- jsonlite::fromJSON(json_response, simplifyVector = FALSE)
      
      outputs_list <- list()
      if (!is.null(parsed_response$outputs)) {
        for (output in parsed_response$outputs) {
          # Since simplifyVector = FALSE, each output should correctly be a list
          name <- output$name
          datatype <- output$datatype
          data <- unlist(output$data) # Unlist if data is deeply nested
          if(datatype == "FP32") {
            data <- as.numeric(data)
          } else if(datatype == "BYTES") {
            data <- as.character(data)
          } else if(datatype == "INT32") {
            data <- as.integer(data)
          }
          # Add more datatype handling as needed
          
          # Reshape the data based on provided shape, example for 2D shape
          if (length(output$shape) == 2) {
            data <- matrix(data, nrow = as.integer(output$shape[1]), byrow = TRUE)
          }
          outputs_list[[name]] <- data
        }
      } else {
        stop("The 'outputs' field is missing in the parsed response.")
      }
      
      return(outputs_list)
    },
    predict = function(input_data) {
      total_samples <- dim(input_data[[1]])[1]
      num_batches <- ceiling(total_samples / .self$batch_size)
      
      # Initialize progress bar
      pb <- txtProgressBar(min = 0, max = num_batches, style = 3)
      
      results <- list()
      
      for (batch_number in seq_len(num_batches)) {
        start_idx <- (batch_number - 1) * .self$batch_size + 1
        end_idx <- min(batch_number * .self$batch_size, total_samples)
        
        # Preparing batch_data
        batch_data <- lapply(input_data, function(arr) {
          # For arrays, slice while preserving dimensional integrity
          arr[start_idx:end_idx, , drop = FALSE]
        })
        
        results <- c(results, list(.self$predict_batch(batch_data)))
        
        # Update progress bar
        setTxtProgressBar(pb, batch_number)
      }
      
      close(pb) # Close the progress bar after the loop completes
      return(aggregate_batches(results))
    },
    aggregate_batches = function(list_of_list_of_arrays) {
      # Initialize an empty list to store aggregated results
      aggregated_results <- list()
      
      # Extract names from the first batch as a reference for aggregation
      reference_names <- names(list_of_list_of_arrays[[1]])
      
      # Loop over each name to aggregate across batches
      for (name in reference_names) {
        # Extract elements for the current name from all batches
        elements <- lapply(list_of_list_of_arrays, function(batch) batch[[name]])
        
        # Determine whether elements are vectors or matrices
        if (is.matrix(elements[[1]])) {
          # Use rbind to concatenate matrices along the first axis
          aggregated_results[[name]] <- do.call(rbind, elements)
        } else {
          # For vectors, concatenate directly and preserve as a matrix for consistency
          concatenated_vector <- do.call(c, elements)
          # Convert to a column matrix, assuming vectors are considered as 1-column matrices
          aggregated_results[[name]] <- matrix(concatenated_vector, ncol = 1)
        }
      }
      
      return(aggregated_results)
    }
  )
)
