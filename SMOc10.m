     %versão simplificada do SMO algoritmo 
    clear;
    clc;
    close all
    global kfunction;
    global gm  
    Index=[];
    X=[];
    y=[];
    alpha=[];
    %ler o data set 
    Data=csvread('dataset1.csv');   

    [N, M]=size(Data); 
    Xdata=Data(:,1:M-1);  
    Ydata=Data(:,M); 
    I=M-1; %número de componentes de cada vetor xn 
    
    %---------- gráfico  de todo o data set 
    figure(1);
    In=find(Ydata==-1);
    Ip=find(Ydata==1);
    plot(Xdata(In,1),Xdata(In,2),'r.','MarkerSize',15)
    hold on
    plot(Xdata(Ip,1),Xdata(Ip,2),'b.','MarkerSize',15);
    hold on
    legend({'-1:Outras Espécies ','+1:Iris Setosa '});
    title('Dataset Iris');
    axis equal
    hold off 
    
   %----------
   
    op=2;   %Data de treino 80% do data set:  op=1   seleciona a partir do inicio 80%; op=2  selecionado aleatoriamente 80%;
    Nt=round(0.8*N);
    if op==1
        p=1:N;
    end
    if op==2  %Data de treino - 80% do Data set -- selecionado aleatoriamente
        p = randperm(N); %returns a row vector containing a random permutation of the integers from 1 to M inclusive
    end
      
    Xt=Xdata(p(1:Nt),:);   
    Yt=Ydata(p(1:Nt),:);  
    
    %Data set para validação - 20% do Data set -- selecionado aleatoriamente 
    Xv=Xdata(p(Nt+1:N),:);
    Yv=Ydata(p(Nt+1:N),:);
   

    %---------- adaptar à configuração do algoritmo SMO
    
    X=Xt';   % Pontos -- organiza-los por coluna: x1; x2; ...;xN
    y=Yt';   % labels correspondentes -- vetor linha: y1,y2,...,yN
    
    [m,N]=size(X);                       % size of data
    alpha=zeros(1,N);                    % alpha variables
    bias=0;                              % initial bias
    it=0;                                % iteration index  
    
    %%DEFINIR PARAMETROS
    C=10;  
    tol=1e-4;
    kfunction='l';  %'l' Linear kernel;  'g' Gaussian kernel;  'p' polynomial kernel 
    sigma=1;
    gm = 1/(2*sigma^2);
    maxit=100;  
    max_passes=1;
    it=0;
    passes=0;
    while (passes < max_passes && it<maxit)                     % number of iterations less than maximum
        it=it+1
        changed_alphas=0;                % number of changed alphas
        for i=1:N                        % for each alpha_i
            Ei=sum(alpha.*y.*K(X,X(:,i)))+bias-y(i);    
            yE=Ei*y(i)
            if (alpha(i)<C && yE<-tol) || (alpha(i)>0 && yE>tol)   % KKT violation
                for j=[1:i-1,i+1:N]        % for each alpha_j not equal alpha_i
                    Ej=sum(alpha.*y.*K(X,X(:,j)))+bias-y(j);
                    ai=alpha(i);         % alpha_i old
                    aj=alpha(j);         % alpha_j old
                    if y(i)==y(j)        % s=y_i y_j=1
                        L=max(0,alpha(i)+alpha(j)-C);
                        H=min(C,alpha(i)+alpha(j));
                    else                 % s=y_i y_j=-1
                        L=max(0,alpha(j)-alpha(i));
                        H=min(C,C+alpha(j)-alpha(i));
                    end
                    if L==H              % skip when L==H
                        continue
                    end
                    eta=2*K(X(:,j),X(:,i))-K(X(:,i),X(:,i))-K(X(:,j),X(:,j));
                    if eta >=0
                        continue
                    end

                    alpha(j)=alpha(j)+y(j)*(Ej-Ei)/eta;   % update alpha_j
                    if alpha(j) > H
                        alpha(j) = H;
                    elseif alpha(j) < L
                        alpha(j) = L;
                    end
                    if norm(alpha(j)-aj) < tol       % skip if no change
                        continue
                    end
                    alpha(i)=alpha(i)-y(i)*y(j)*(alpha(j)-aj);   % find alpha_i
                    bi = bias - Ei - y(i)*(alpha(i)-ai)*K(X(:,i),X(:,i))...
                        -y(j)*(alpha(j)-aj)*K(X(:,j),X(:,i));
                    bj = bias - Ej - y(i)*(alpha(i)-ai)*K(X(:,i),X(:,j))...
                        -y(j)*(alpha(j)-aj)*K(X(:,j),X(:,j));   
                    if 0<alpha(i) && alpha(i)<C
                        bias=bi;
                    elseif 0<alpha(j) && alpha(j)<C
                        bias=bj;
                    else
                        bias=(bi+bj)/2;
                    end
                    changed_alphas=changed_alphas+1;  % one more alpha changed
                end
            end
        end
        if changed_alphas==0             % no more changed alpha, quit
            passes=passes+1;
        else
            passes=0;
        end
       
    end % end of iteration
    
     Index=find(abs(alpha)>10^(-3));     % indecies of non-zero alphas
     alpha=alpha(Index);                  % find non-zero alphas
     Xsv=X(:,Index);                      % find support vectors
     ysv=y(Index);                        % their corresponding indecies
     nsv=length(ysv);  

    
     it
     passes


    %%%  calcular bias
    bias=0;
     for i=1:nsv   
        bias=bias+(ysv(i)-sum(ysv.*alpha.*K(Xsv,Xsv(:,i))));
    end
    bias=bias/nsv;

     %--- 
    fprintf('SVM - Vetores de suporte: \n');
    fprintf('  n:    alpha_n         Xsv                   Y_n   \n');
    for j=1:length(Index)
       fprintf('%4i    %.4f      x=[%.4f, %.4f]     %4d \n', Index(j), alpha(j), Xsv(1,j), Xsv(2,j), ysv(j)); 
      
    end
    
    %classificador
    Xx=Xv';
    [m n]=size(Xx);
    for i=1:n
     Yp(i)=sign(sum(alpha.*ysv.*K(Xsv,Xx(:,i)))+bias);    % Yp - classe prevista pelo classificador
    end
    
   %%%
   ErrDv=sum(abs(Yp-Yv'));  % out-sample error == erro com o data de validação
   fprintf('\n out-sample error: %.4e ', ErrDv);
    
   %---- gráfico  do data set de treino 
    figure
    In=find(Yt==-1);
    Ip=find(Yt==1);
    h(1)=plot(Xt(In,1),Xt(In,2),'r.','Markersize',15);
    hold on
    h(2)=plot(Xt(Ip,1),Xt(Ip,2),'b.','MarkerSize',15);
    hold on
    %-- gráfico dos support vectors
    h(3)=plot(X(1,Index),X(2,Index),'ko','MarkerSize',8);
    hold on
    %---- opcional
    flag=1;  % flag=1 faz também o gráfico do  data set de validação; flag=0  não faz o gráfico do  data set de validação
    if flag
        In=find(Yv==-1);
        Ip=find(Yv==1);
        h(4)=plot(Xv(In,1),Xv(In,2),'ro','Markersize',4);
        hold on
        h(5)=plot(Xv(Ip,1),Xv(Ip,2),'bo','MarkerSize',4);
        hold on
    end
        
    %--- faz o gráfico da fronteira de decisão
    d = 0.1;
    [x1Grid,x2Grid] = meshgrid(min(Xt(:,1)):d:max(Xt(:,1)),...
    min(Xt(:,2)):d:max(Xt(:,2)));
    xGrid = [x1Grid(:), x2Grid(:)];
  
    %classificador  
    Xx=xGrid';
    [m n]=size(Xx);
    for i=1:n
     Yp(i)=sum(alpha.*ysv.*K(Xsv,Xx(:,i)))+bias;    % Yp - valor dado pelo classificador
    end
    contour(x1Grid,x2Grid,reshape(Yp,size(x1Grid)),[0 0], 'k');  %'ShowText','on'
     
 
    %----
    title('SVM dual com SMO  + fronteira de decisão');
    if ~flag
       legend(h,{'-1:Outras Especies-treino','+1:Setosa-treino','Vetores Suporte'});    
    else  
       legend(h,{'-1:Outras Espécies-treino','+1:Setosa-treino','Vetores Suporte', '-1:Outras Espécies-validação','+1:Setosa-validação'});
        
    end
    axis equal
    hold on

  
    %
    
    