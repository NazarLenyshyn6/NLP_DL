
DL Lab 2 Software Architecture:

Architecture:

* params Folder
    * Shapes - module which contains function to deal with input shapes
    * Weights - module which contains function to initialise weights
* propagation
    * forward_propagation - module which contains objects which  perform forward propagation
    * back_propagation - module which contains object which perform back propagation 
* predictio - module which contains function to perform predictions
* model
    * model_builder - module which contains function to build model
* Factories 
    * propagation_factorie-modeul which contains factories for createing objects from propagation folder
    * model_factories - module which contains factored for createin model
* Configuration:
    * config_entitry - module which will contains data oriented class for specific  configs
    * Configuration manager - module which will manage all config_entities
* Utils - module which contains all commonly repeated functions
* Logger - model which contains configuration of logger
* Config - module which contains all required configuration parameters