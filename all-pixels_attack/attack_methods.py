# -*- coding: utf-8 -*-

import time
import numpy as np
import deap.creator, deap.tools, deap.base, deap.algorithms, deap.cma

IMG_SIZES = (224, 224, 3)

class general_method:
    def __init__(self, target_model, target_img, target_class=None, constraint=0.05):
        self.constraint = constraint
        self.target_model = target_model
        self.target_img = target_img
        self.target_class = target_class
        
        self.predicted = self.target_model.predict(self.target_img)[0].argsort()[-5:][::-1]
        
        deap.creator.create("FitnessMax", deap.base.Fitness, weights=(1.0,))
        deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMax)
        self.toolbox = deap.base.Toolbox()
    
    
    '''
    check if a candidate is valid
    '''
    def is_feasible(self, individual):
        return np.max(np.abs(individual)) <= self.constraint
    
    '''
    add perturbation to target image
    '''
    def add_adv(self, adv):
        if len(adv.shape) < 3:
            adv = adv.reshape(IMG_SIZES)
        
        # clip if the magnitude of perturbation excesses constraint
        if not self.is_feasible(adv):
            adv = np.clip(adv, -self.constraint, self.constraint)
            
        adv_img = self.target_img / 255.0 + adv
        adv_img = np.clip(adv_img, 0, 1)
        adv_img = adv_img * 255.0
        
        # make sure that all pixels of image are integer number
        adv_img = np.rint(adv_img)
        return adv_img

    '''
    evaluate the fitnesses of all individuals on population
    '''   
    def evaluate_batch(self, individuals):
        if isinstance(individuals, list):
            individuals = np.array(individuals)
        m = individuals.shape[0]
        if not self.is_feasible(individuals):
            individuals = np.clip(individuals, -self.constraint, self.constraint)
            
        adv_img = self.target_img / 255.0 + individuals
        adv_img = np.clip(adv_img, 0, 1)
        adv_img = adv_img * 255.0
        # make sure that all pixels of image are integer number
        adv_img = np.rint(adv_img)
        
        pred = self.target_model.predict(adv_img)
        
        max_top5_confidence = np.zeros(m)
        for predicted_idx in self.predicted:
            max_top5_confidence = np.maximum(max_top5_confidence, pred[np.arange(m), predicted_idx])
            pred[np.arange(m), predicted_idx] = 0     
        
        max_confidence_non_target = np.max(pred, axis=1)
    
        return np.log(max_confidence_non_target) - np.log(max_top5_confidence)
        
class simpleEA(general_method):
    def __init__(self, target_model, target_img, target_class=None, constraint=0.05):
        super().__init__(target_model=target_model, target_img=target_img, target_class=target_class, constraint=constraint)

        self.toolbox.register("individual", deap.tools.initRepeat, 
                         container=deap.creator.Individual, func=lambda: np.random.uniform(-self.constraint, self.constraint), 
                         n=IMG_SIZES[0]*IMG_SIZES[1]*IMG_SIZES[2])

        self.toolbox.register("population", deap.tools.initRepeat, list, self.toolbox.individual)

    
    def evaluate_batch(self, individuals):
        if isinstance(individuals, list):
            individuals = np.array(individuals)
        individuals = individuals.reshape((-1, IMG_SIZES[0], IMG_SIZES[1], IMG_SIZES[2]))
        return super().evaluate_batch(individuals)
        
        
    def evaluate(self, individuals):
        if isinstance(individuals, list):
            individuals = np.array(individuals)
        individuals = individuals.reshape((-1, IMG_SIZES[0], IMG_SIZES[1], IMG_SIZES[2]))
        return (super().evaluate_batch(individuals)[0],)
    '''
    def evol(self, num_gen=99):
        self.toolbox.register("evaluate", self.evaluate)
    
        pop = self.toolbox.population(n=32)
        self.hof = deap.tools.HallOfFame(64,  similar=np.array_equal)
        stats = deap.tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        start = time.time()

        self.toolbox.register("mate", deap.tools.cxOnePoint)
        self.toolbox.register("mutate", deap.tools.mutGaussian, mu=0, sigma=0.01, indpb=1)
        self.toolbox.register("select", deap.tools.selTournament, tournsize=3)

        pop, logbook = deap.algorithms.eaMuCommaLambda(pop, self.toolbox, mu=10, lambda_=20, cxpb=0.5, mutpb=0.5, ngen=num_gen, stats=stats, halloffame=self.hof, verbose=True)
        end = time.time()
        print("Runing time: ", end - start)
    '''
    
    def evol(self, num_gen, mu=9, lambda_=18, verbose=True):
        self.toolbox.register("mate", deap.tools.cxOnePoint)
        self.toolbox.register("mutate", deap.tools.mutGaussian, mu=0, sigma=0.05, indpb=0.05)
        self.toolbox.register("select", deap.tools.selTournament, tournsize=3)

        self.hof = deap.tools.HallOfFame(1, similar=np.array_equal)
        stats = deap.tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        logbook = deap.tools.Logbook()

        population = self.toolbox.population(n=mu)
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = self.evaluate_batch(invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)

        self.hof.update(population)

        for gen in range(1, num_gen + 1):
            offspring = deap.algorithms.varOr(population, self.toolbox, lambda_=lambda_, cxpb=0.5, mutpb=0.5)

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.evaluate_batch(invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = (fit,)

            population[:] = self.toolbox.select(offspring, mu)

            self.hof.update(population)
            
            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(population), **record)
            if verbose:
                print(logbook.stream)

            if self.hof[0].fitness.values[0] >= 0:
                return gen
        return -1
        
    def generate_adv_image(self):
        adv_img = self.add_adv(self.hof[0])
        return adv_img
        
        

class CMA_ES(general_method):
    def __init__(self, target_model, target_img, target_class=None, base_vectors=None, constraint=0.05, init_point=None):
        super().__init__(target_model=target_model, target_img=target_img, target_class=target_class, constraint=constraint)

        self.flag_toolbox = False
        
        self.base_vectors = None
        self.set_base_vectors(base_vectors)
        
        self.setup_toolbox()
        
        self.set_init_point(init_point)
        
    def set_base_vectors(self, base_vectors=None, num_base_vectors=None):
        self.flag_toolbox = True
        if base_vectors is None:
            if num_base_vectors is not None:
                self.num_base_vectors = num_base_vectors 
                #print("self.num_base_vectors", self.num_base_vectors)
            else:
                self.num_base_vectors = 128
            self.base_vectors = np.random.uniform(-self.constraint, self.constraint, (IMG_SIZES[0]* IMG_SIZES[1]* IMG_SIZES[2], self.num_base_vectors))
        else:
            self.base_vectors = base_vectors
            self.num_base_vectors = self.base_vectors.shape[0]
            
        self.setup_toolbox()
   
    def set_init_point(self, init_point):
        self.init_point = self.toolbox.individual()
        if init_point is not None:
            for i, v in enumerate(init_point):
                self.init_point[i] = v
   
    def setup_toolbox(self):
        self.toolbox.register("individual", deap.tools.initRepeat, 
                         container=deap.creator.Individual, func=lambda: np.random.uniform(-1.0/self.num_base_vectors, 1.0/self.num_base_vectors), 
                         n=self.num_base_vectors)
                         
        self.toolbox.register("population", deap.tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate)
   
   
    def get_adv(self, individual):
        return np.sum(np.multiply(self.base_vectors, individual), axis=1).reshape(IMG_SIZES)
    
    def evaluate(self, individual):
        adv = self.get_adv(individual)
        
        return super().evaluate(adv)

    def evaluate_batch(self, individuals):
        m = len(individuals)
        advs = np.zeros(shape=(m, IMG_SIZES[0], IMG_SIZES[1], IMG_SIZES[2]))
        for i, individual in enumerate(individuals):
            advs[i] = self.get_adv(individual)
            
        return super().evaluate_batch(advs)
    '''
    def evol(self, num_gen=20, sigma=0.1):
        begin = time.time()
        self.hof = deap.tools.HallOfFame(self.num_base_vectors, similar=np.array_equal)
        stats = deap.tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        strategy = deap.cma.Strategy(self.toolbox.individual(), sigma=sigma)
        #self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("generate", strategy.generate, ind_init=deap.creator.Individual)
        self.toolbox.register("update", strategy.update)
        if not self.flag_toolbox:
            raise("The toolbox has not been recompiled!")
        
        pop, log = deap.algorithms.eaGenerateUpdate(self.toolbox, ngen=num_gen, halloffame=self.hof, stats=stats)
        end = time.time()
        print("Runing time: ", end - begin)
    '''
    
    def evol(self, max_gen, sigma=0.2, verbose=True, num_base_vectors=128):
        self.setup_toolbox()
        self.set_base_vectors(num_base_vectors=num_base_vectors)
        strategy = deap.cma.Strategy(self.toolbox.individual(), sigma=sigma)
        self.toolbox.register("generate", strategy.generate, deap.creator.Individual)
        self.toolbox.register("update", strategy.update)

        self.hof = deap.tools.HallOfFame(1, similar=np.array_equal)
        stats = deap.tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = deap.tools.Logbook()
        logbook.header = ['gen', 'nevals'] + stats.fields

        for gen in range(1, max_gen + 1, 1):
            # Generate a new population
            population = self.toolbox.generate()

            # Evaluate the individuals
            fitnesses = self.evaluate_batch(population)#
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = (fit,)

            # update hall of fame
            self.hof.update(population)

            # Update the strategy with the evaluated individuals
            self.toolbox.update(population)

            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(population), **record)
            if verbose:
                print(logbook.stream)

            if self.hof[0].fitness.values[0] > 0:
                return gen
        return -1
        
        
    def generate_adv_image(self):
        adv = np.zeros(IMG_SIZES)
        for i, t in enumerate(self.hof[0]):
            adv += t * self.base_vectors[i]
        adv_img = self.add_adv(adv)
        
        return adv_img