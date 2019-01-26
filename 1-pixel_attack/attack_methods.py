import numpy as np
import random   
import deap.creator, deap.tools, deap.base, deap.algorithms, deap.cma

IMG_SIZES = (32, 32, 3)

class CIFAR10_attacker:
    def __init__(self, target_model, target_img, target_class=None, num_pixels=1):
        self.num_pixels = num_pixels
        self.target_model = target_model
        self.target_img = target_img
        self.target_class = target_class
        self.predicted_class = np.argmax(self.target_model.predict_one(self.target_img))
        
        deap.creator.create("FitnessMax", deap.base.Fitness, weights=(1.0,))
        deap.creator.create("Individual", np.ndarray, fitness=deap.creator.FitnessMax)
        self.toolbox = deap.base.Toolbox()
        
        self.toolbox.register("individual", deap.tools.initRepeat, 
                         container=deap.creator.Individual, func=lambda: random.random(), n=self.num_pixels*5)

        self.toolbox.register("population", deap.tools.initRepeat, list, self.toolbox.individual)
    
    
    def add_adv(self, adv):
        adv = np.clip(adv, 0, 1)
        
        adv_img = np.copy(self.target_img)
        
        # each modification of 1 pixel includes (x, y, dr, dg, gb) which are position and channel values of the pixel we want to modify
        for i in range(0, self.num_pixels * 5, 5):
            x = int(adv[i] * 31)
            y = int(adv[i+1] * 31)
            dr, dg, db = (adv[i+2:i+5] * 255).astype(np.uint8)
            adv_img[x, y, 0] = dr
            adv_img[x, y, 1] = dg
            adv_img[x, y, 2] = db
        return adv_img
    
    
    def evaluate_batch(self, individuals):
        individuals = np.clip(individuals, 0, 1)
        m = len(individuals)
        adv_imgs = np.zeros((m, IMG_SIZES[0], IMG_SIZES[1], IMG_SIZES[2]))
        for i, individual in enumerate(individuals):
            adv_imgs[i] = self.add_adv(individual)
        
        pred = self.target_model.predict_batch(adv_imgs)
        #pred_log = np.log(pred)
        success = False
        if self.target_class is None: # non-targeted attack
            label_confidences = pred[np.arange(m), self.predicted_class]
            pred[np.arange(m), self.predicted_class] = 0
            attack_confidences = np.max(pred, axis=1)
            for i in range(m):
                if attack_confidences[i] > label_confidences[i]:
                    success = True
            return attack_confidences - label_confidences, success
            #return np.log(attack_confidences) - np.log(label_confidences)
        else: # targeted attack
            attack_confidences = pred[np.arange(m), self.target_class]
            pred[np.arange(m), self.target_class] = 0
            other_confidences = np.max(pred, axis=1)
            return np.log(attack_confidences) - np.log(other_confidences)
            
            
    def cma_es_attack(self, num_gen, sigma=0.5, verbose=True):
        strategy = deap.cma.Strategy(self.toolbox.individual(), sigma=sigma, lambda_=64)
        self.toolbox.register("generate", strategy.generate, deap.creator.Individual)
        self.toolbox.register("update", strategy.update)
        
        self.hof = deap.tools.HallOfFame(1, similar=np.array_equal)
        stats = deap.tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = deap.tools.Logbook()
        
        for gen in range(1, num_gen + 1):
            population = self.toolbox.generate()

            fitnesses, success = self.evaluate_batch(population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = (fit,)
                     
            
            self.hof.update(population)
            
            self.toolbox.update(population)
            
            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(population), **record)
            if verbose:
                print(logbook.stream)
             
            if success:
                return True
        return False
            
            
    def eaMuCommaLambda_attack(self, num_gen, mu=32, lambda_=64, cxpb=0.5, mutpb=0.5, verbose=True):
        self.toolbox.register("mate", deap.tools.cxOnePoint)
        self.toolbox.register("mutate", deap.tools.mutGaussian, mu=0, sigma=0.1, indpb=0.5)
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
        fitnesses, success = self.evaluate_batch(invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)

        self.hof.update(population)
        
        for gen in range(1, num_gen + 1):
            offspring = deap.algorithms.varOr(population, self.toolbox, lambda_, cxpb, mutpb)
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses, success = self.evaluate_batch(invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = (fit,)
                
            population[:] = self.toolbox.select(offspring, mu)
            
            self.hof.update(population)
            
            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(population), **record)
            if verbose:
                print(logbook.stream)

            if success:
                return True
        return False
                
                
    def DE_attack(self, num_gen, verbose=True):
        CR_rate = 0.3
        F = 0.5
        MU = 400

        self.toolbox.register("select", deap.tools.selRandom, k=3)
        
        population = self.toolbox.population(n=MU);
        self.hof = deap.tools.HallOfFame(1, similar=np.array_equal)
        stats = deap.tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = deap.tools.Logbook()
        #logbook.header = "gen", "evals", "std", "min", "avg", "max"

        # Evaluate the individuals
        fitnesses, _ = self.evaluate_batch(population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = (fit,)

        for gen in range(1, num_gen + 1):
            next_gen = []
            for k, agent in enumerate(population):
                a,b,c = self.toolbox.select(population)
                y = self.toolbox.clone(agent)
                        
                CR = np.random.binomial(n=1, p=CR_rate, size=agent.shape)
                CR_bar = 1 - CR
                
                y[:] = CR_bar * agent + CR * (a + F * (b - c))
                
                next_gen.append(y)
            fitnesses, success = self.evaluate_batch(next_gen)
    
            for k, agent in enumerate(population):
                if agent.fitness.values[0] <= fitnesses[k]:
                    population[k] = next_gen[k]
                    population[k].fitness.values = (fitnesses[k],)
    
            self.hof.update(population)
            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, evals=len(population), **record)
            if verbose:
                print(logbook.stream)
                
            if success:
                return True
        return False