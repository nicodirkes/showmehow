## R Server for UMBridge Protocol
library(httpuv)
library(httr2)
library(jsonlite)
library(future)
library(promises)
suppressMessages(library(RobustGaSP)) # to avoid printing to the terminal



Model <- R6::R6Class("Model",
  public = list(
    name = NULL,

    initialize = function(name) {
      self$name <- name
    },

    get_input_sizes = function() {
      stop(sprintf("You need to implement this method in %s.", class(self)[1]))
    },
    get_output_sizes = function() {
      stop(sprintf("You need to implement this method in %s.", class(self)[1]))
    },
    evaluate = function(parameters, config = list()) {
      stop(sprintf("Method called but not implemented in %s.", class(self)[1]))
    },
    gradient = function(out_wrt, in_wrt, parameters, sens, config = list()) {
      stop(sprintf("Method called but not implemented in %s.", class(self)[1]))
    },
    apply_jacobian = function(out_wrt, in_wrt, parameters, vec, config = list()) {
      stop(sprintf("Method called but not implemented in %s.", class(self)[1]))
    },
    apply_hessian = function(out_wrt, in_wrt1, in_wrt2, parameters, sens, vec, config = list()) {
      stop(sprintf("Method called but not implemented in %s.", class(self)[1]))
    },


    supports_evaluate = function() {
      FALSE
    },
    supports_gradient = function() {
      FALSE
    },
    supports_apply_jacobian = function() {
      FALSE
    },
    supports_apply_hessian = function() {
      FALSE
    }
  )
)

supported_models <- function(url) {
  stopifnot(typeof(url) == "character")
  response <- httr2::request(url) |>
    httr2::req_url_path_append("Info") |>
    httr2::req_perform() |>
    httr2::resp_body_json(check_type = FALSE)

  if (response$protocolVersion != 1.0) {
    stop("Model has unsupported protocol version!")
  }
  return(response$models)
}

HTTPModel <- R6::R6Class("HTTPModel",
  inherit = Model,
  public = list(
    url = NULL,
    
    initialize = function(url, name) {
      super$initialize(name)
      self$url <- url
      if (!is.element(name, supported_models(url))) {
        stop(sprintf("Model %s not supported by server! Supported models are: %s", 
                     name,
                     paste(supported_models(url), collapse = ", ")))
      }

      input <- list(name = name)
      response <- httr2::request(self$url) |>
        httr2::req_url_path_append("ModelInfo") |>
        httr2::req_body_json(input) |>
        httr2::req_perform() |>
        httr2::resp_body_json(check_type = FALSE)

      self$supports_evaluate <- response$support$Evaluate
      self$supports_gradient <- response$support$Gradient
      self$supports_apply_jacobian <- response$support$ApplyJacobian
      self$supports_apply_hessian <- response$support$ApplyHessian
    },

    get_input_sizes = function(config = list()) {
      input <- list(name = self$name, config = config)
      response <- httr2::request(self$url) |>
        httr2::req_url_path_append("InputSizes") |>
        httr2::req_body_json(input) |>
        httr2::req_perform() |>
        httr2::resp_body_json(check_type = FALSE)
      return(response$inputSizes)
    },

    get_output_sizes = function(config = list()) {
      input <- list(name = self$name, config = config)
      response <- httr2::request(self$url) |>
        httr2::req_url_path_append("OutputSizes") |>
        httr2::req_body_json(input) |>
        httr2::req_perform() |>
        httr2::resp_body_json(check_type = FALSE)
      return(response$outputSizes)
    },

    supports_evaluate = function() {
      return(self$supports_evaluate)
    },

    supports_gradient = function() {
      return(self$supports_gradient)
    },

    supports_apply_jacobian = function() {
      return(self$supports_apply_jacobian)
    },

    supports_apply_hessian = function() {
      return(self$supports_apply_hessian)
    },

    check_input_is_list_of_lists = function(parameters) {
      if (!is.list(parameters) || !all(sapply(parameters, is.list))) {
        stop("Parameters must be a list of lists!")
      }
    },

    evaluate = function(parameters, config = list()) {
      if (!self$supports_evaluate()) {
        stop("Evaluation not supported by model!")
      }
      self$check_input_is_list_of_lists(parameters)

      inputParams <- list(name = self$name, input = parameters, config = config)
      response <- httr2::request(self$url) |>
        httr2::req_url_path_append("Evaluate") |>
        httr2::req_body_json(inputParams) |>
        httr2::req_perform() |>
        httr2::resp_body_json(check_type = FALSE)

      if (!is.null(response$error)) {
        stop(sprintf("Model returned error of type %s: %s", response$error$type, response$error$message))
      }
      return(response$output)
    },

    gradient = function(out_wrt, in_wrt, parameters, sens, config = list()) {
      if (!self$supports_gradient()) {
        stop("Gradient not supported by model!")
      }
      self$check_input_is_list_of_lists(parameters)

      inputParams <- list(name = self$name, outWrt = out_wrt, inWrt = in_wrt, input = parameters, sens = sens, config = config)
      response <- httr2::request(self$url) |>
        httr2::req_url_path_append("Gradient") |>
        httr2::req_body_json(inputParams) |>
        httr2::req_perform() |>
        httr2::resp_body_json(check_type = FALSE)

      if (!is.null(response$error)) {
        stop(sprintf("Model returned error of type %s: %s", response$error$type, response$error$message))
      }
      return(response$output)
    },

    apply_jacobian = function(out_wrt, in_wrt, parameters, vec, config = list()) {
      if (!self$supports_apply_jacobian()) {
        stop("ApplyJacobian not supported by model!")
      }
      self$check_input_is_list_of_lists(parameters)

      inputParams <- list(name = self$name, outWrt = out_wrt, inWrt = in_wrt, input = parameters, vec = vec, config = config)
      response <- httr2::request(self$url) |>
        httr2::req_url_path_append("ApplyJacobian") |>
        httr2::req_body_json(inputParams) |>
        httr2::req_perform() |>
        httr2::resp_body_json(check_type = FALSE)

      if (!is.null(response$error)) {
        stop(sprintf("Model returned error of type %s: %s", response$error$type, response$error$message))
      }
      return(response$output)
    },

    apply_hessian = function(out_wrt, in_wrt1, in_wrt2, parameters, sens, vec, config = list()) {
      if (!self$supports_apply_hessian()) {
        stop("ApplyHessian not supported by model!")
      }
      self$check_input_is_list_of_lists(parameters)

      inputParams <- list(name = self$name, outWrt = out_wrt, inWrt1 = in_wrt1, inWrt2 = in_wrt2, input = parameters, sens = sens, vec = vec, config = config)
      response <- httr2::request(self$url) |>
        httr2::req_url_path_append("ApplyHessian") |>
        httr2::req_body_json(inputParams) |>
        httr2::req_perform() |>
        httr2::resp_body_json(check_type = FALSE)

      if (!is.null(response$error)) {
        stop(sprintf("Model returned error of type %s: %s", response$error$type, response$error$message))
      }
      return(response$output)
    }
  )
)

serve_models <- function(models, port = 4242, max_workers = 1) {
  # Set up asynchronous processing
  future::plan(multisession, workers = max_workers)

  error_response <- function(type, message, status) {
    response <- list(error = list(type = type, message = message))
    return(list(
      status = status,
      headers = list("Content-Type" = "application/json"),
      body = jsonlite::toJSON(response, auto_unbox = TRUE)
    ))
  }

  model_not_found_response <- function(model_name) {
    return(error_response("ModelNotFound", sprintf("Model %s not found! The following are available: %s", model_name, paste(lapply(models, function(model) model$name), collapse = ", ")), 400))
  }


  get_model_from_name <- function(name) {
    for (model in models) {
      if (model$name == name) {
        return(model)
      }
    }
    return(NULL)
  }

  # Web server routes
  app <- list(
    call = function(request) {
      path <- request$PATH_INFO
      method <- request$REQUEST_METHOD
      response <- switch(path,
        "/Evaluate" = if (method == "POST") {
          promises::as.promise(
            future::future({
              body <- jsonlite::fromJSON(request$rook.input$read_lines(), simplifyVector = FALSE)
              model <- get_model_from_name(body$name)
              if (is.null(model)) {
                return(model_not_found_response(body$name))
              }
              if (!model$supports_evaluate()) {
                return(error_response("UnsupportedFeature", "Evaluate not supported by model!", 400))
              }
              parameters <- body$input
              config <- ifelse(is.null(body$config), list(), body$config)

              input_sizes <- model$get_input_sizes(config)
              output_sizes <- model$get_output_sizes(config)

              # Check if parameter dimensions match model input sizes
              if (length(parameters) != length(input_sizes)) {
                return(error_response("InvalidInput", "Number of input parameters does not match model number of model inputs!", 400))
              }
  
              for (i in seq_along(parameters)) {
                if (length(parameters[[i]]) != input_sizes[[i]]) {
                  return(error_response("InvalidInput", sprintf("Input parameter %d has invalid length! Expected %d but got %d.", i, input_sizes[[i]], length(parameters[[i]])), 400))
                }
              }

              output <- model$evaluate(parameters, config)

              # Check if output is a list of lists
              if (!is.list(output)) {
                return(error_response("InvalidOutput", "Model output is not a list of lists!", 500))
              }
              if (!all(sapply(output, is.list))) {
                return(error_response("InvalidOutput", "Model output is not a list of lists!", 500))
              }

              # Check if output dimensions match model output sizes
              if (length(output) != length(output_sizes)) {
                return(error_response("InvalidOutput", "Number of output vectors returned by model does not match number of model outputs declared by model!", 500))
              }
              for (i in seq_along(output)) {
                if (length(output[[i]]) != output_sizes[[i]]) {
                  return(error_response("InvalidOutput", sprintf("Output vector %d has invalid length! Model declared %d but returned %d.", i, output_sizes[[i]], length(output[[i]])), 500))
                }
              }

              return(list(
                status = 200,
                headers = list("Content-Type" = "application/json"),
                body = jsonlite::toJSON(list(output = output), auto_unbox = TRUE)
              ))
            })
          )
        },
        # "/Gradient" = if (method == "POST") {
        #   promises::as.promise(
        #     future::future({
        #       body <- jsonlite::fromJSON(request$rook.input$read_lines())
        #       model <- get_model_from_name(body$name)
        #       if (is.null(model)) {
        #         return(model_not_found_response(body$name))
        #       }
        #       if (!model$supports_gradient()) {
        #         return(error_response("UnsupportedFeature", "Gradient not supported by model!", 400))
        #       }
        #       out_wrt <- body$outWrt + 1 # Adjusting for R's 1-based indexing
        #       in_wrt <- body$inWrt + 1 # Adjusting for R's 1-based indexing
        #       parameters <- body$input
        #       sens <- body$sens
        #       config <- ifelse(is.null(body$config), list(), body$config)

        #       input_sizes <- model$get_input_sizes(config)
        #       output_sizes <- model$get_output_sizes(config)

        #       # Check if parameter dimensions match model input sizes
        #       if (length(parameters) != length(input_sizes)) {
        #         return(error_response("InvalidInput", "Number of input parameters does not match model number of inputs!", 400))
        #       }
        #       for (i in seq_along(parameters)) {
        #         if (length(parameters[[i]]) != input_sizes[[i]]) {
        #           return(error_response("InvalidInput", sprintf("Input parameter %d has invalid length! Expected %d but got %d.", i, input_sizes[[i]], length(parameters[[i]])), 400))
        #         }
        #       }
        #       # Check if out_wrt is between 1 and number of outputs
        #       if (out_wrt < 1 || out_wrt > length(output_sizes)) {
        #         return(error_response("InvalidInput", sprintf("Invalid outWrt index! Expected between 1 and number of outputs, but got %d", out_wrt), 400))
        #       }
              
        #       # Check if in_wrt is between 1 and number of inputs
        #       if (in_wrt < 1 || in_wrt > length(input_sizes)) {
        #         return(error_response("InvalidInput", sprintf("Invalid inWrt index! Expected between 1 and number of inputs, but got %d", in_wrt), 400))
        #       }
              
        #       # Check if sensitivity vector length matches model output outWrt
        #       if (length(sens) != output_sizes[[out_wrt]]) {
        #         return(error_response("InvalidInput", sprintf("Sensitivity vector sens has invalid length! Expected %d but got %d.", output_sizes[[out_wrt]], length(sens)), 400))
        #       }

        #       output <- model$gradient(out_wrt - 1, in_wrt - 1, parameters, sens, config) # Adjusting for R's 1-based indexing

        #       # Check if output is a list
        #       if (!is.list(output)) {
        #         return(error_response("InvalidOutput", "Model output is not a list!", 500))
        #       }

        #       # Check if output dimension matches model input size inWrt
        #       if (length(output) != input_sizes[[in_wrt]]) {
        #         return(error_response("InvalidOutput", sprintf("Output vector has invalid length! Model declared %d but returned %d.", input_sizes[[in_wrt]], length(output)), 500))
        #       }

        #       return(list(
        #         status = 200,
        #         headers = list("Content-Type" = "application/json"),
        #         body = jsonlite::toJSON(list(output = output), auto_unbox = TRUE)
        #       ))
        #     })
        #   )
        # },
        # "/ApplyJacobian" = if (method == "POST") {
        #   promises::as.promise(
        #     future::future({
        #       body <- jsonlite::fromJSON(request$rook.input$read_lines())
        #       model <- get_model_from_name(body$name)
        #       if (is.null(model)) {
        #         return(model_not_found_response(body$name))
        #       }
        #       if (!model$supports_apply_jacobian()) {
        #         return(error_response("UnsupportedFeature", "ApplyJacobian not supported by model!", 400))
        #       }
        #       out_wrt <- body$outWrt + 1 # Adjusting for R's 1-based indexing
        #       in_wrt <- body$inWrt + 1 # Adjusting for R's 1-based indexing
        #       parameters <- body$input
        #       vec <- body$vec
        #       config <- ifelse(is.null(body$config), list(), body$config)
              
        #       input_sizes <- model$get_input_sizes(config)
        #       output_sizes <- model$get_output_sizes(config)

        #       # Check if parameter dimensions match model input sizes
        #       if (length(parameters) != length(input_sizes)) {
        #         return(error_response("InvalidInput", "Number of input parameters does not match model number of inputs!", 400))
        #       }
        #       for (i in seq_along(parameters)) {
        #         if (length(parameters[[i]]) != input_sizes[[i]]) {
        #           return(error_response("InvalidInput", sprintf("Input parameter %d has invalid length! Expected %d but got %d.", i, input_sizes[[i]], length(parameters[[i]])), 400))
        #         }
        #       }

        #       # Check if outWrt is between 1 and number of outputs
        #       if (out_wrt < 1 || out_wrt > length(output_sizes)) {
        #         return(error_response("InvalidInput", sprintf("Invalid outWrt index! Expected between 1 and number of outputs, but got %d", out_wrt), 400))
        #       }

        #       # Check if inWrt is between 1 and number of inputs
        #       if (in_wrt < 1 || in_wrt > length(input_sizes)) {
        #         return(error_response("InvalidInput", sprintf("Invalid inWrt index! Expected between 1 and number of inputs, but got %d", in_wrt), 400))
        #       }

        #       # Check if vector length matches model input inWrt
        #       if (length(vec) != input_sizes[[in_wrt]]) {
        #         return(error_response("InvalidInput", sprintf("Vector vec has invalid length! Expected %d but got %d.", input_sizes[[in_wrt]], length(vec)), 400))
        #       }

        #       output <- model$apply_jacobian(out_wrt - 1, in_wrt - 1, parameters, vec, config) # Adjusting for R's 1-based indexing

        #       # Check if output is a list
        #       if (!is.list(output)) {
        #         return(error_response("InvalidOutput", "Model output is not a list!", 500))
        #       }

        #       # Check if output dimension matches model output size outWrt
        #       if (length(output) != output_sizes[[out_wrt]]) {
        #         return(error_response("InvalidOutput", sprintf("Output vector has invalid length! Model declared %d but returned %d.", output_sizes[[out_wrt]], length(output)), 500))
        #       }

        #       return(list(
        #         status = 200,
        #         headers = list("Content-Type" = "application/json"),
        #         body = jsonlite::toJSON(list(output = output), auto_unbox = TRUE)
        #       ))
        #     })
        #   )
        # },
        # "/ApplyHessian" = if (method == "POST") {
        #   promises::as.promise(
        #     future::future({
        #       body <- jsonlite::fromJSON(request$rook.input$read_lines())
        #       model <- get_model_from_name(body$name)
        #       if (is.null(model)) {
        #         return(model_not_found_response(body$name))
        #       }
        #       if (!model$supports_apply_hessian()) {
        #         return(error_response("UnsupportedFeature", "ApplyHessian not supported by model!", 400))
        #       }
        #       out_wrt <- body$outWrt + 1 # Adjusting for R's 1-based indexing
        #       in_wrt1 <- body$inWrt1 + 1 # Adjusting for R's 1-based indexing
        #       in_wrt2 <- body$inWrt2 + 1 # Adjusting for R's 1-based indexing
        #       parameters <- body$input
        #       sens <- body$sens
        #       vec <- body$vec
        #       config <- ifelse(is.null(body$config), list(), body$config)

        #       input_sizes <- model$get_input_sizes(config)
        #       output_sizes <- model$get_output_sizes(config)

        #       # Check if parameter dimensions match model input sizes
        #       if (length(parameters) != length(input_sizes)) {
        #         return(error_response("InvalidInput", "Number of input parameters does not match model number of inputs!", 400))
        #       }
        #       for (i in seq_along(parameters)) {
        #         if (length(parameters[[i]]) != input_sizes[[i]]) {
        #           return(error_response("InvalidInput", sprintf("Input parameter %d has invalid length! Expected %d but got %d.", i, input_sizes[[i]], length(parameters[[i]])), 400))
        #         }
        #       }

        #       # Check if outWrt is between 1 and number of outputs
        #       if (out_wrt < 1 || out_wrt > length(output_sizes)) {
        #         return(error_response("InvalidInput", sprintf("Invalid outWrt index! Expected between 1 and number of outputs, but got %d", out_wrt), 400))
        #       }

        #       # Check if inWrt1 is between 1 and number of inputs
        #       if (in_wrt1 < 1 || in_wrt1 > length(input_sizes)) {
        #         return(error_response("InvalidInput", sprintf("Invalid inWrt1 index! Expected between 1 and number of inputs, but got %d", in_wrt1), 400))
        #       }

        #       # Check if inWrt2 is between 1 and number of inputs
        #       if (in_wrt2 < 1 || in_wrt2 > length(input_sizes)) {
        #         return(error_response("InvalidInput", sprintf("Invalid inWrt2 index! Expected between 1 and number of inputs, but got %d", in_wrt2), 400))
        #       }              
        #       output <- model$apply_hessian(out_wrt - 1, in_wrt1 - 1, in_wrt2 - 1, parameters, sens, vec, config) # Adjusting for R's 1-based indexing

        #       # Check if output is a list
        #       if (!is.list(output)) {
        #         return(error_response("InvalidOutput", "Model output is not a list!", 500))
        #       }

        #       # Check if output dimension matches model output size outWrt
        #       if (length(output) != output_sizes[[out_wrt]]) {
        #         return(error_response("InvalidOutput", sprintf("Output vector has invalid length! Model declared %d but returned %d.", output_sizes[[out_wrt]], length(output)), 500))
        #       }

        #       return(list(
        #         status = 200,
        #         headers = list("Content-Type" = "application/json"),
        #         body = jsonlite::toJSON(list(output = output), auto_unbox = TRUE)
        #       ))
        #     })
        #   )
        # },
        "/InputSizes" = if (method == "POST") {
          promises::as.promise(
            future::future({
              body <- jsonlite::fromJSON(request$rook.input$read_lines())
              model <- get_model_from_name(body$name)
              if (is.null(model)) {
                return(model_not_found_response(body$name))
              }
              config <- ifelse(is.null(body$config), list(), body$config)
              input_sizes <- model$get_input_sizes(config)
              return(list(
                status = 200,
                headers = list("Content-Type" = "application/json"),
                body = jsonlite::toJSON(list(inputSizes = input_sizes), auto_unbox = TRUE)
              ))
            })
          )
        },
        "/OutputSizes" = if (method == "POST") {
          promises::as.promise(
            future::future({
              body <- jsonlite::fromJSON(request$rook.input$read_lines())
              model <- get_model_from_name(body$name)
              if (is.null(model)) {
                return(model_not_found_response(body$name))
              }
              config <- ifelse(is.null(body$config), list(), body$config)
              output_sizes <- model$get_output_sizes(config)
              return(list(
                status = 200,
                headers = list("Content-Type" = "application/json"),
                body = jsonlite::toJSON(list(outputSizes = output_sizes), auto_unbox = TRUE)
              ))
            })
          )
        },
        "/ModelInfo" = if (method == "POST") {
          promises::as.promise(
            future::future({
              body <- jsonlite::fromJSON(request$rook.input$read_lines())
              model <- get_model_from_name(body$name)
              if (is.null(model)) {
                return(model_not_found_response(body$name))
              }
              support <- list(
                Evaluate = model$supports_evaluate(),
                Gradient = model$supports_gradient(),
                ApplyJacobian = model$supports_apply_jacobian(),
                ApplyHessian = model$supports_apply_hessian()
              )
              return(list(
                status = 200,
                headers = list("Content-Type" = "application/json"),
                body = jsonlite::toJSON(list(support = support), auto_unbox = TRUE)
              ))
            })
          )
        },
        "/Info" = if (method == "GET") {
          promises::as.promise(
            future::future({
              protocol_version <- 1.0
              models_list <- lapply(models, function(model) model$name)
              body = jsonlite::toJSON(list(protocolVersion = as.double(protocol_version), models = models_list), auto_unbox = TRUE)
              return(list(
                status = 200,
                headers = list("Content-Type" = "application/json"),
                body = body
              ))
            })
          )
        },
        return(list(
          status = 404,
          headers = list("Content-Type" = "application/json"),
          body = jsonlite::toJSON(list(error = list(type = "NotFound", message = "Endpoint not found")), auto_unbox = TRUE)
        ))
      )

      return(response)
    }
  )

  # Start the server
  httpuv::runServer("0.0.0.0", port, app)
}






# Define a surrogate model for testing
SurrogateModel <- R6::R6Class("SurrogateModel",
  inherit = Model,
  public = list(
    scalar_gp = NULL,
    initialize = function(name, filepath) {
      super$initialize(name)
      self$scalar_gp <- readRDS(filepath)
    },

    get_input_sizes = function(config = list()) {
      return(list(2))
    },

    get_output_sizes = function(config = list()) {
      return(list(1))
    },

    evaluate = function(parameters, config = list()) {
      model_input <- matrix(as.numeric(parameters[[1]]), nrow = 1, ncol = 2)
      #Perform prediction using the scalar_gp
      model_output <- predict(self$scalar_gp, model_input)
      return(list(list(model_output$mean)))
    },

    supports_evaluate = function() {
      return(TRUE)
    }
  )
)

# Create and serve the test model

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)
stopifnot("Missing Arguements. \nCorrect Usage: Rscript um_evaluate.R <path> <model_filename> <model_name>" = length(args) == 2)
model_filename <- args[1]
model_name <- args[2]

model_path <- file.path(model_filename)
gaussian_process_model <- SurrogateModel$new(model_name, model_path)
serve_models(list(gaussian_process_model))