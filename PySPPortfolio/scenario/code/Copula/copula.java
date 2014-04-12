/**
 * Vector Generator with Copulas
 *
 * @author J. Ch. Strelen
 * @version 1.00 28.2.2008
 *
 */
 
  import java.io.*;
  import java.lang.Math;
  import java.math.*;
  import java.lang.System.*;
 
 
  
   class Zufalls_Zahlen   // random numbers
{
   static private long faktor=16807;
   static private long modul=1024*1024*2048-1;
   static private double v = 1 / (double)modul;
   private long y;
   
   public int[] saat = {1, 
      1550655590, 766698560, 1645116277, 1154667137, 1961833381, 460575272,  
      1497719440, 901595110, 354421844, 1800697079, 821079954, 1918430133,  
      1736896562, 634729389, 23246132, 1008653149, 1689665648, 1628735393,  
      550023088, 1267216792, 314116942, 2095003524, 356484144, 2140958617,  
      1116852912, 1701937813, 171817360, 360646631, 1652205397, 605448579,  
      1223969344, 1821072732, 1237280745, 2125855022, 935058038, 1151620124,  
      1970906347, 66562942, 1503754591, 2007872839, 1870986444, 1375265396,  
      470646700, 500432100, 347147950, 929595364, 800443847, 5715478,  
      2032092669, 996425830, 1884232020, 1821061493, 248900140, 905130946,  
      1307421505, 1091346202, 1140141037, 1244153851, 611793617, 808278095,  
      2106046913, 1226683843, 90274853, 2147466840, 2140522509, 1146667727,  
      1530171233, 319146380, 2077765218, 789950931, 632682054, 1683083109,  
      351704670, 171230418, 1986612391, 1411714374, 872179784, 801917373,  
      144283230, 1949917822, 113201992, 1965781805, 679060319, 630187802,  
      1298843779, 1565131991, 50366922, 144513213, 207666443, 13354949,  
      631135695, 958408264, 494931978, 1150735880, 1640573652, 1315013967,  
      1254156333, 587564032, 1912367727, 2138169990, 2059764593, 115613893,  
      131630606, 1386707532, 2081362360, 1428465736, 1170668648, 931140599,  
      197473249, 1381732824, 925311926, 576725369, 198434005, 1296038143,  
      653782169, 1503907840, 33491376, 238345926, 1366925216, 1549695660,  
      1793656969, 1701980729, 1883864564, 251608257, 642486710, 1115145546,  
      1021484058   
   };
      // Seed for streams 1 bis 128,  2^24 r.n. apart,
      // but 128: Only  2^24-1 r.n.
   
   public Zufalls_Zahlen(int stromnummer) {    // Constructor with stream
      y = saat[stromnummer-1];
   }
    	
     
   public Zufalls_Zahlen() {     // Constructor
      y = 1;
   }
   
   public void Strom(int stromnummer) {    //  set stream number
      y = saat[stromnummer-1];
   }
   
   // uniform random numbers:
   
   public double U_0_1()
   {  y=(y*faktor) % modul;
      return ((double)y) * v;
   }
}
   
  
  class VectorGenerator {
  	
  	// Debug ddd = new Debug();
  	
  	public int D, Ds, n;
  	public float u_gen[], z_gen[]; 
  	
    int K, ms, shift, m, d, mm, ds, nn, h, D_minus_shift, i;
                 // i==1 for vectors, i==2 for time series vectors but the 1st
    float power_K, power_K_2, delta;
    double theta = (double) Math.sqrt(5)*16;    //for hash function
  	int j_values[][][];
    int coll_ptr[][];
    int Beg[][];
    int tt_index[][];
    int j[];
    float tt_f[][];
    float tt_s[][];
    float Z_sorted[][];
    int trf[];
    Zufalls_Zahlen ZZ;
    		
  	StringBuffer buf;
    	
    	InputStreamReader isr = null;  	
    	char c;
    	int ic;
    	
    	VectorGenerator(){
    		ZZ = new Zufalls_Zahlen(1);
    		System.out.println(
    		  "This program is copyright the original author and the University");
    		System.out.println(
    		  "of Bonn, and is published here under the GNU General Public License.");
    		System.out.println(
    		  "(See http://www.fsf.org/licenses/licenses.html)");
    	}
    	
    	public void set_stream(int ZZStream) {
    		ZZ.Strom(ZZStream);
    	}
   	
    	public float InF() {
    	  float x = 0;
    	  try{	
    		buf = new StringBuffer(50); 
    		ic = isr.read();
    		c=(char)ic;
    		// System.out.println(ic + " " + c + " " + (int)c);
    		while((c  == ' ') | (ic==10) ){  // ' '  newline
    		    // buf.append(' '); 
    			ic = isr.read();
    			c=(char)ic;
    			// System.out.println(ic + " " + c + " " + (int)c);
    		}
    		if (ic!=(-1))
    			buf.append(c); 
    		else System.out.println("End of File found");
    		ic = isr.read();
    		c=(char)ic;
    		// System.out.println(ic + " " + c + " " + (int)c);
    		while( ( ic!=(-1) ) & ( c!=' ' ) & (ic != 10)) { // eof ' ' newline
    		   buf.append(c);
    		   ic = isr.read();
    		   c=(char)ic;
    			// System.out.println(ic + " " + c + " " + (int)c);
    		}
    		x = Float.parseFloat(buf.toString());
    		// System.out.println("gelesen: " + x); 
    	  }   
    	  catch(IOException e) {
    	  	System.out.println("IOException ");
     	  }	
    	  return x;
    	}
    	
    	public void buildCopula(String fname){
    		
    		try{
    			isr = new InputStreamReader(new FileInputStream(fname));
    		}   
    		catch(IOException e) {
    			System.out.println("IOException ");
    		}    		
    		
    		K = (int) InF();  System.out.println("K= "+K);
    		D = (int) InF();  System.out.println("D= "+D);
    		n = (int) InF();  System.out.println("n= "+n);
    		ms = (int) InF();  System.out.println("ms= "+ms);
    		m = ms +n;  System.out.println("m= "+m);
    		shift = (int) InF();  System.out.println("shift=D'= "+shift);
    		Ds = shift;
    		D_minus_shift=D-shift;
    		i=1; //next generated vector is the first
    		power_K = (float) Math.pow(K,D-1);
    		power_K_2 = (float) Math.pow(K,D-2);
    		delta = (float) (1.0/K);
    		
    		j_values = new int[D-1][m+1][D];
    		coll_ptr = new int[D-1][m+1];
    		Beg = new int[D-1][m+1];
    		tt_index = new int[D-1][n];
    		tt_f = new float[D-1][n];
    		tt_s = new float[D-1][n];
    		u_gen =  new float[D];
        	j = new int[D];
    		
    		     /* alle Indices sind um -1 verschoben außer m, i.e. collision pointers */
    		     /* d_1 = d-1, nn_1 = nn-1, ds_1 = ds-1 */
    		for (int d_1=0; d_1 <= D-2; ++d_1 ) {
    			for (int mm=1; mm <= m; ++mm ){
    				for (int ds_1=0; ds_1 <= D-1; ++ds_1 ){
    					j_values[d_1][mm][ds_1] = (int) InF();
    				}
    			}
    		}
    		for (int d_1=0; d_1 <= D-2; ++d_1 ) {
    			for (int mm=1; mm <= m; ++mm ){
    				coll_ptr[d_1][mm] = (int) InF();
    			}
    		}
    		for (int d_1=0; d_1 <= D-2; ++d_1 ) {
    			for (int mm=1; mm <= m; ++mm ){
    				Beg[d_1][mm] = (int) InF();
    			}
    		}
    		for (int d_1=0; d_1 <= D-2; ++d_1 ) {
    			for (int nn_1=0; nn_1 <= n-1; ++nn_1 ){
    				tt_index[d_1][nn_1] = (int) InF();
    			}
    		}
    		for (int d_1=0; d_1 <= D-2; ++d_1 ) {
    			for (int nn_1=0; nn_1 <= n-1; ++nn_1 ){
    				tt_f[d_1][nn_1] =  InF();
    			}
    		}
    		
    		for (int d_1=0; d_1 <= D-2; ++d_1 ) {
    			for (int nn_1=0; nn_1 <= n-1; ++nn_1 ){
    				tt_s[d_1][nn_1] =  InF();
    			}
    		}
    		
    		/* Zwausg
    		System.out.println("j_values");
    		for (int ds_1=0; ds_1 <= D-1; ++ds_1 ){
    			for (int d_1=0; d_1 <= D-2; ++d_1 ) {
    				for (int mm=1; mm <= m; ++mm ){
    					d=d_1+1;  ds=ds_1+1;
    					System.out.println
    					("["+d+"]["+mm+"]["+ds+"]= "+j_values[d_1][mm][ds_1] );
    				}
    			}
    		}
    		System.out.println("coll_ptr");
    		for (int d_1=0; d_1 <= D-2; ++d_1 ) {
    			for (int mm=1; mm <= m; ++mm ){
    					d=d_1+1;
    					System.out.println
    					("["+d+"]["+mm+"]= "+coll_ptr[d_1][mm] );
    			}
    		}
    		System.out.println("Beg");
    		for (int d_1=0; d_1 <= D-2; ++d_1 ) {
    			for (int mm=1; mm <= m; ++mm ){
    				d=d_1+1; 
    					System.out.println
    					("["+d+"]["+mm+"]= "+Beg[d_1][mm] );
    			}
    		}
    		System.out.println("tt_index");
    		for (int d_1=0; d_1 <= D-2; ++d_1 ) {
    			for (int nn_1=0; nn_1 <= n-1; ++nn_1 ){
    				d=d_1+1; nn=nn_1+1;
    					System.out.println
    					("["+d+"]["+nn+"]= "+tt_index[d_1][nn_1] );
    			}
    		}
    		System.out.println("tt_f");
    		for (int d_1=0; d_1 <= D-2; ++d_1 ) {
    			for (int nn_1=0; nn_1 <= n-1; ++nn_1 ){
    				d=d_1+1; nn=nn_1+1;
    					System.out.println
    					("["+d+"]["+nn+"]= "+tt_f[d_1][nn_1] );
    			}
    		}
    		
    		System.out.println("tt_s");
    		for (int d_1=0; d_1 <= D-2; ++d_1 ) {
    			for (int nn_1=0; nn_1 <= n-1; ++nn_1 ){
    				d=d_1+1; nn=nn_1+1;
    					System.out.println
    					("["+d+"]["+nn+"]= "+tt_s[d_1][nn_1] );
    			}
    		} 
    		*/
    		gen_u();  // u_gen is now initialised
    		
    	} // end buildCopula
    	
    	
        public void buildEmpDistr(String emp_file, int [] par_trf){
        	trf = new int[Ds];	
        	trf = par_trf;
        	
    		try{
    			isr = new InputStreamReader(new FileInputStream(emp_file));
    		}   
    		catch(IOException e) {
    			System.out.println("IOException ");
    		}
    		
    		Z_sorted = new float[Ds][n];
    		z_gen =  new float[Ds]; 
    		
    		for (int d_1=0; d_1 <= Ds-1; ++d_1 ) {
    			d=d_1+1; 
    			for (int nn_1=0; nn_1 <= n-1; ++nn_1 ){
    				nn=nn_1+1;
    				Z_sorted[d_1][nn_1] = InF();
    			}
    		}
    		
    		/* System.out.println("Z_sorted");                 //!!!
    		for (int d_1=0; d_1 <= D-1; ++d_1 ) {
    			for (int nn_1=0; nn_1 <= n-1; ++nn_1 ){
    				d=d_1+1; nn=nn_1+1;
    					System.out.println
    					("["+d+"]["+nn+"]= "+Z_sorted[d_1][nn_1] );
    			}
    		 } */
        }
        
        void gen_u () {
        	int u1_gen_bar, ttp;
        	double sigma;
        	boolean ungleich;
        	int j2, u2_check, u2_gen_bar, j_d, u_d_check, u_d_gen_bar;
        	float u_gen_, u_power_K, summ, summ_old=0, f_value=0, u_gen_rest, 
        		u2_gen_rest, u_gen_f_dm1, u_d_gen_rest;
        	
        	// Index 0: 
        	d=1;
        	if ((d>D_minus_shift) || (i==1)){ // first vector or 
        			// no AR process but random vector or d > D-Ds
	        	u_gen[0] = (float) ZZ.U_0_1();
	        	// System.out.println( " u_gen_0 " + u_gen[0]);
	        	u1_gen_bar = (int) Math.ceil(u_gen[0]*K);   //  delta = 1/K
	        	j[0] = u1_gen_bar;
        	} else {                    // time series, 2., 3., ... vector
        		u_gen[0] = u_gen[shift];
        		j[0] =  j[shift]; 
        	}
        	
        	d=2;
        	// Calculate hash table entry adress h ..................................v
        	// Given: d and tuple j(1:d-1)
        	// Result: Hash table entry address h.  
        	//   h>0 -> tuple found,
        	//   h==0 -> tuple not in table, hence f == 0 and s == 0
        
        	// Calculate hash adress h ..................................vv
        	sigma=0;
        	for (int dss_1=0; dss_1 <= d-2; ++dss_1 ) {
            	sigma = (sigma + j[dss_1])/K;
        	}
        	h = (int) Math.floor( ms*( sigma*theta - Math.floor(sigma*theta) ) ) + 1;
        	// End calculate hash adress  ..............................^^
        
        	// while j != j[d-1;0:d-2] follow collision chain:
        	ungleich = false;
        	for (int dss_1=0; dss_1 <= d-2; ++dss_1 ) {
        		if (j[dss_1]!= j_values[d-2][ h][ dss_1] )
        		   ungleich = true;
        	}
         	while (h>0 & ungleich)  {
         		h = coll_ptr[d-2][h];
         		ungleich = false;
        		for (int dss_1=0; dss_1 <= d-2; ++dss_1 ) {
        			if (j[dss_1]!= j_values[d-2][ h][ dss_1] )
        		   		ungleich = true;
        		}
         	}
         	// End calculate hash table entry adress  ..............................^
        	
        	
        	
        	ttp = Beg[d-2][h]; 
        	if ((d>D_minus_shift) || (i==1)){ // first vector or 
        			// no AR process but random vector or d > D-Ds
	        	u_gen_ = (float) ZZ.U_0_1();
	        	// System.out.println( " u_gen_ " + u_gen_);
	        	u_power_K = u_gen_ * power_K;
	        	summ = 0;
	        	j2 = 0;
	      		while (summ < u_power_K) {
	        		/* if ttp<1 | ttp>n | d<2 | d>D   // error
	            		ttp
	            		d
	        		   end; */
	        		j2 = tt_index[d-2][ttp-1];   
	        		summ_old = summ;
	        		summ = tt_s[d-2][ttp-1];
	        		f_value = tt_f[d-2][ttp-1];   //  = f2(u1_gen_bar, j2);
	        		ttp = ttp+1;
	      		}
	    
	    		u2_check = j2-1;
	    		u2_gen_bar = j2;
	    		u_gen_rest = u_gen_ - summ_old/power_K;  
	    		u2_gen_rest = u_gen_rest*power_K_2 / f_value;   
	    		 		// f_value = f2(u1_gen_bar, u2_gen_bar);
	    		u_gen[1] = u2_check*delta + u2_gen_rest;
	    		j[1] = u2_gen_bar;
    		} else {                    // time series, 2., 3., ... vector
    			j2 = j[1+shift];
    			while (j2 != tt_index[d-2][ttp-1] ){
    				ttp++;
    			}
    			f_value = tt_f[d-2][ttp-1];   //  = f2(u1_gen_bar, j2);
    			u_gen[d-1] = u_gen[d-1+shift];
    			j[d-1] = j[d-1+shift]; 
    		}
    		
    		for ( d=3; d <= D; ++d ) {
    			// Calculate hash table entry adress h ..................................v
        		// Given: d and tuple j(1:d-1)
        		// Result: Hash table entry address h.  
        		//   h>0 -> tuple found,
        		//   h==0 -> tuple not in table, hence f == 0 and s == 0
        
        		// Calculate hash adress h ..................................vv
        		sigma=0;
        		for (int dss_1=0; dss_1 <= d-2; ++dss_1 ) {
            		sigma = (sigma + j[dss_1])/K;
        		}
        		h = (int) Math.floor( ms*( sigma*theta - Math.floor(sigma*theta) ) ) + 1;
        		   	// System.out.println( " h " + h);
        		// End calculate hash adress  ..............................^^
        
        		// while j != j[d-1;0:d-2] follow collision chain:
        		ungleich = false;
        		for (int dss_1=0; dss_1 <= d-2; ++dss_1 ) {
        				// System.out.println( "j[dss_1] " + j[dss_1]);
        				// System.out.println( "j_values[dss_1] " + j_values[d-2][ h][ dss_1]);
        			if (j[dss_1]!= j_values[d-2][ h][ dss_1] )
        			   ungleich = true;
        		}
         		while (h>0 & ungleich)  {
         			h = coll_ptr[d-2][h]; //   System.out.println( " h " + h); //Zwausg
         			ungleich = false;
        			for (int dss_1=0; dss_1 <= d-2; ++dss_1 ) {
        				// System.out.println( "j[dss_1] " + j[dss_1]);
        				// System.out.println( "j_values[dss_1] " + j_values[d-2][ h][ dss_1]);
        				if (j[dss_1]!= j_values[d-2][ h][ dss_1] )
        		   			ungleich = true;
        			}
         		}
         		// End calculate hash table entry adress  ..............................^
        		
        		ttp = Beg[d-2][h];   // System.out.println( " ttp " + ttp); //Zwausg
                
        		if ((d>D_minus_shift) || (i==1)){ // first vector or 
        						// no AR process but random vector or d > D-Ds
	    			u_gen_ = (float) ZZ.U_0_1();
	    			// System.out.println( " u_gen_ " + u_gen_);
	    			summ = 0;
	    			j_d = 0;
	    			u_gen_f_dm1 = u_gen_* f_value;  // = f_d-1(u1_gen_bar,...,u_d-1_gen_bar);
	    			
				    while (summ < u_gen_f_dm1) {
				        j_d = tt_index[d-2][ttp-1];
				        summ_old = summ;
				        summ = tt_s[d-2][ttp-1];    //= s(d, u1_gen_bar,...,u_d-1_gen_bar, j_d);
				        f_value = tt_f[d-2][ttp-1]; //= f(d, u1_gen_bar,...,u_d-1_gen_bar, j_d);
				        ttp = ttp+1;  //   System.out.println( " ttp " + ttp); //Zwausg
				    }
				    u_d_check = j_d-1;
				    u_d_gen_bar = j_d;
				    u_gen_rest = delta*(u_gen_f_dm1 - summ_old); 
				    u_d_gen_rest = u_gen_rest / f_value;     
				         // =  f(d,u1_gen_bar,..., u_d_gen_bar);
							// System.out.println
							//( "u_d_ckeck ... "+u_d_check+" "+u_d_gen_bar+" "+u_gen_rest+" "+u_d_gen_rest); //Zwausg
								//System.out.println
							//( "delta ... "+delta+" "+u_gen_f_dm1+" "+summ_old+" "); //Zwausg
				    u_gen[d-1] = u_d_check*delta + u_d_gen_rest;
				    
				    j[d-1] = u_d_gen_bar; 
				 } else {                    // time series, 2., 3., ... vector
				    int jd = j[d-1+shift];
    				while (jd != tt_index[d-2][ttp-1] ){
    					ttp++;
    				}
    				f_value = tt_f[d-2][ttp-1];   //  = fd(u1_gen_bar, jd);
    				u_gen[d-1] = u_gen[d-1+shift];
    				j[d-1] = j[d-1+shift]; 
				 }  
    			
    		}  // end for d
    		
    		
        } // end public void gen_u ()
        
        public void gen_u_vector (){
        	i=1;
        	gen_u();
        }
        
        public void gen_u_ar (){
        	i=2;
        	gen_u();
        }
        
        public void gen_z () {
        	float un;
        	int i,d,ds_1;
        	for (int d_1=0; d_1 <= D-1; ++d_1 ) {
        		d = d_1+1;
        		ds_1 = d_1 % Ds;
        		un = u_gen[d_1]*n;
        		i = (int) Math.ceil(un)-1;
        		if (i <=0){
        			if (i<0) { i=0; }
        			z_gen[ds_1] = Z_sorted[ds_1][0];	// smallest generated value
        		}
        		else {
        		  if (trf[ds_1]<=1) {
        			z_gen[ds_1] = Z_sorted[ds_1][i-1] 
        			        + (un-i)*(Z_sorted[ds_1][i]-Z_sorted[ds_1][i-1]);
        		  } else {
        			z_gen[ds_1] = Z_sorted[ds_1][i];   // i in pwlCopula different!
        		  }
        		}
        	}
        }
  	
  	
  }
 
public class VGen {
	
	
    public static void main(String[] args) throws IOException {
    	
    	BufferedReader din = new BufferedReader(
           new InputStreamReader(System.in),15);
           // str = din.readLine();
			//      System.out.println(str);
      		// a=Integer.parseInt(str);
      		
      		// D=D.valueOf(str);   Umwandlung String --> Double (Hüllkl.)
    		//  x=D.doubleValue(); Umwandlung Double --> double
    		
    		//
    	
    	int D, Ds, same_inv_trf, ZZStream, d;
    	String str, emp_file, fname;
    	long t0; 
    	
    	
    	VectorGenerator VG = new VectorGenerator();
    		   
    	// Build copula:
        System.out.println( "Which copula file (without .cop)? "  );	
    	fname = din.readLine();
        VG.buildCopula(fname + ".cop");
    	D = VG.D;
    	Ds = VG.Ds;
           
        // Build empirical distributions:
    	int [] trf = new int[Ds];
    	//  System.out.println( "Which empirical distribution file? "  );	
    	emp_file = fname + ".emp";  // din.readLine();
    	same_inv_trf=0;
    	for (int d_1=0; d_1 <= Ds-1; ++d_1 ) {
    		d=d_1+1;
    		if((same_inv_trf==0)|(d==1)) {
    			System.out.println( "Dimension " + d + ", Inverse transform");
    			System.out.println( "   with linearly interpolated empirical CDF (1), e.g. for real numbers "   );
    			System.out.println( "   with empirical CDF (2), e.g. for integer numbers "   );
    			str = din.readLine(); trf[d_1]  = Integer.parseInt(str);
    			while ((trf[d_1]<1)|(trf[d_1]>2)){
    				System.out.println( "1 or 2 !!! " );
    				str = din.readLine(); trf[d_1]  = Integer.parseInt(str);
    			}
    			if (d==1) {
	    			System.out.println( "Same inverse transform for all componens? (0,1) "  );
	    			str = din.readLine(); same_inv_trf = Integer.parseInt(str);
	    		} else { 
	    		}
    		} else {
    		    trf[d_1] = trf[0];
    		}
    	}   		 
        VG.buildEmpDistr(emp_file,trf);
        
        // Generate some u-vectors and z_vectors:
    	System.out.println( "How many generated vectors? "  );
    	str = din.readLine(); int n_gen = Integer.parseInt(str);
        while (n_gen>0) {
        
	    	System.out.println( "Which Random Number Stream? (1...128) "  );	
	    	str = din.readLine(); ZZStream=Integer.parseInt(str);
	    	VG.set_stream(ZZStream);
	          
	        float mean[]= new float[D];
	        for (int d_1=0; d_1<=Ds-1; d_1++){
	        	mean[d_1]=0;
	        } 
	        
	        t0 = java.lang.System.currentTimeMillis();
	        for(int i=1; i<=n_gen; ++i) {
	        	if (i==1)
	        		VG.gen_u_vector();
	        	else
	        		VG.gen_u_ar();	 
	        	if ((i==2) || (i==3)) {   //(i <=10) {
	        		System.out.println( "u_gen= " );
	        		for (int d_1=0; d_1<=D-1; d_1++){
	        			System.out.print ( VG.u_gen[d_1] + " ");
	        			if ( (d_1+1) % 10 == 0 ){
	        				System.out.println (  " ");
	        				System.out.print (  "   ");
	        			}
	        		}
	        		System.out.println (  " ");	
	        	}
	        	
	        	VG.gen_z();
	        	for (int d_1=0; d_1<=Ds-1; d_1++){
	        		mean[d_1] += VG.z_gen[d_1];
	        	}	
	        	if  ((i==2) || (i==3)) {   // (i<=10) {
	        		System.out.println( "z_gen= " );
		        	for (int d_1=0; d_1<=Ds-1; d_1++){
		        		System.out.print ( VG.z_gen[d_1] + " ");
		        		if ( (d_1+1) % 10 == 0 ){
		        			System.out.println (  " ");
		        			System.out.print (  "   ");
		        		}
		        		System.out.println (  " ");
		        	}
	        	}
	    	}
	    	
	        System.out.println( "CPU time for generation: " + 
	               (java.lang.System.currentTimeMillis() - t0) + " msec");
	        System.out.println( "mean(z_gen) = " );
	        for (int d_1=0; d_1<=Ds-1; d_1++){
	        	System.out.print ( (mean[d_1]/n_gen) + " ");
	        	if ( (d_1+1) % 10 == 0 ){
	        		System.out.println (  " ");
	        		System.out.print (  "   ");
	        	}
	        }
	        
	    	System.out.println( "More generated vectors? How many? "  );
	    	str = din.readLine(); n_gen = Integer.parseInt(str);
	    }
        
    }
}
 
 
