import java.io.*;
import java.util.*;

/* All methods in Perceptron are static. Has functionality for
 * creating Example arraylists from files, creating weight vectors
 * from training sets, testing the accuracy of a weight vector on 
 * a validation set, and all necessary sub functions */

public class perceptron{
  
  public static final int numberFeatures = 29328; //the total number of features
  
  /*
   * Test the error rate of classifying Examples in validate using weight vector
   * w. Print out total success rate. 
   */ 
  
  public static Example paValidation(Example w, ArrayList<Example> validate, double mu){
    
    ArrayList<Example> validationSet = validate;
    Example e = w;
    double totalExamples = 0;
    double totalCorrect = 0;
    int n = validationSet.size();
    for(int i=0;i<n;i++){
      Example current = validationSet.get(i);
      double correctAnswer = current.classification;
      double machineAnswer = evaluate(e,current);
      if(correctAnswer == machineAnswer){totalCorrect ++;}
      totalExamples++;
    }
    System.out.println(totalCorrect + " correct out of " + totalExamples);
    return e;
  }
  
  /* Generate an ArrayList of examples from the polarity file 
   * in filename */
  
  public static ArrayList<Example> createFromFile(String filename){
    
    ArrayList<Example> a = new ArrayList<Example>();
    
    try{
      BufferedReader bufferedReader = new BufferedReader(new FileReader(filename));
      String line = null;
      
      
      while ((line = bufferedReader.readLine()) != null) {
        Example e = new Example();
        String curr = line;
        
        int sIndex = curr.indexOf(" ");
        String rawClass = curr.substring(0,sIndex);
        double cl = Double.parseDouble(rawClass);
        
        e.setClassification((int)cl);
        curr=curr.substring(sIndex+1);
        
        
        boolean t = true;
        while(t){
          String colon = ":";
          String space = " ";
          
          int colonIndex = curr.indexOf(colon);
          int spaceIndex = curr.indexOf(space);
          
          String rawVal = "";
          
          String rawFeat = curr.substring(0,colonIndex);
          
          if(spaceIndex>0){
            rawVal = curr.substring(colonIndex+1,spaceIndex);
            curr = curr.substring(spaceIndex+1);
          }
          else{
            rawVal = curr.substring(curr.indexOf(colon)+1);
            t=false;
          }
          
          double val = Double.parseDouble(rawVal);
          double feat = Double.parseDouble(rawFeat);
          
          e.addPair(feat,val);
        }
        a.add(e);
      }
    }
    catch (Exception e)
    {
      System.err.println("File input error "+ e);
    }
    return a;
  }
  
  /* The main function call for generating an weight vector from training set 
   * built from tFile with learning rate mu and training iterations it. Returns
   * the weight vector and prints error rate information */
  
  public static Example iterativePerceptron(String tFile, String vFile, double mu, int its){
    
    ArrayList<Example> S = createFromFile(tFile);
    ArrayList<Example> V = createFromFile(vFile);
    
    Example w = new Example(numberFeatures);
    
    for(int i =0; i< its; i++){
      int wrongCount =0;
      for(int j=0;j<S.size();j++){
        
        Example myEx = S.get(j);
        
        double dotProduct = dotProduct(w,myEx);
        double bias = w.getBias();
        double y = myEx.getClassification();
        
        if(y*(dotProduct) <= 0){
          //System.out.println("Improving..");
          wrongCount++;
          w=improveW(w,myEx,mu,y, dotProduct);
        }
      }
      int rightCount = 201-wrongCount;
      System.out.println(rightCount+ " out of 201 correct on training.");
      paValidation(w,V,mu);
      
    }
    
    return w;
  }
  
  /* Called when w misclassifies x. Use the perceptron update formula to adjust
   * the weights of w using learning rate y. */
  
  public static Example improveW(Example w, Example x, double mu, double y, double dp){
    Example e = w;
    double max = 0;
    int [] keyArray = x.keyArray();
    int n = keyArray.length;
    for(int i=0;i<n;i++){
      // update max(||xi||) if necessary
      double euc = x.eucSize;
      if(euc>max){max = euc;}
      int f = keyArray[i];
      double xValue = x.getValueByFeature(f);
      double wValue = w.getValueByFeature(f);
      //update feature-value pair using update rules
      w.setPair(f, wValue + xValue*(y-dp)*mu);
    }
    //update bias of e and return e
    double b = e.bias;
    double newB = b + mu * (y-dp) * Math.pow(max,2);
    e.setBias(newB);
    return e;
  }
  // classify x using weight vector w
  
  public static double evaluate(Example w, Example x){
    double dp = dotProduct(x,w);
    double val = dp;
    if(val>0){return 1;}else{return -1;}
  }
  
  /* = the dot product of x1 and x2, used by improveW() 
   * and perceptronAlgorithm() */
  
  public static double dotProduct(Example x1, Example x2){
    double sumVal = 0;
    // put all features used by x1 in an array
    int [] keyArray = x1.keyArray();
    int n = keyArray.length;
    for(int i=0;i<n;i++){
      //retrieve each feature
      int indexer = keyArray[i];
      // multiply feature f in x1 by f in x2
      sumVal = sumVal + x1.getValueByFeature(indexer)*x2.getValueByFeature(indexer);
    }
    return sumVal;
  }
}