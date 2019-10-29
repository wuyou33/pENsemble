clear all
clc

% load susytraintest; name = 'EnseSusy'; nFolds=10000;
% ninput=18;
% noutput=2;
% sourcedata=data;


% load Hepmass2m; name = 'EnseHepm'; nFolds=4000;
% ninput=28;
% noutput=2;
% sourcedata=data;


% load rlcps; name = 'EnseRlcps'; nFolds=10000;
% ninput=9;
% noutput=2;
% sourcedata=data;


% load rfid; name = 'EnseRfid'; nFolds=560;
% ninput=3;
% noutput=4;
% sourcedata=data;


% load electricitypricing; ninput=8; name = 'EnseElec'; nFolds=90;
% noutput=2;
% sourcedata=data;

% load occupancy; ninput=5; name = 'EnseOccup'; nFolds=16;
% noutput=2;
% sourcedata=data;

% load weather; name = 'EnseWeather'; nFolds=36;
% ninput=8;
% noutput=2;
% sourcedata=data;

% load kdd10persenshuff3; name = 'EnseKdd'; nFolds=1000;
% ninput=41;
% noutput=2;
% sourcedata=data;

load sea; name = 'EnseSea'; nFolds=200;
ninput=3;
noutput=2;
sourcedata=data;

% load datahyp; name = 'EnseHyp'; nFolds=240;
% ninput=4;
% noutput=2;
% M=noutput;
% sourcedata=data;


% load datamnist; name = 'EnseMnist'; nFolds=140;
% a=find(max(data)==1);
% data=data(:,a);
% ninput=size(data,2)-10;
% noutput=10
% sourcedata=data;
% sourcedata=sourcedata(1:100,:); nFolds=10;


decreasingfactor=0.5;
threshold=0.01;
confidenceinterval=0.001;
lambdaD=0.001;
lambdaW=0.005;
partial=0;
subset=2;
LR=0.2;
RF=0.01;
local=1;
p=1;
RSMnew=0;
RSMdev=0;
%spambase=[spambasedatatrain;spambasedatatest]

[nData,nData1]=size(sourcedata);
%spambase=[spambasedatatrain;spambasedatatest]
%             [creditcardoutput,pendigits_Data]=modify_dataset_zero_class(sourcedata);
%[wineInputs1]=normal_class(processeddataset(:,1:end-1));

%           data=[sourcedata(:,1:end-1) creditcardoutput];


%data=[CB(:,1:2) creditcardoutput] ;

ensembleoutput=[];
inputpruning=1;
ensemblepruning1=1;
ensemblepruning2=1;
ensemblesize=[];

A1=[];
B=[];
C=[];
D=[];
E=[];
F=[];
chunk_size=floor(nData/nFolds);
buffer=[];
counter=0;
ensemble=0;
t = 0;
Residual_error = [];


for k=1:chunk_size:nData
    %% testing
    t = t+1;
    if (k+chunk_size-1) > nData
        Data = data(k:nData,:);    %Tn = T(n:nTrainingData,:);
    else
        Data = data(k:(k+chunk_size-1),:);   % Tn = T(n:(n+Block-1),:);
    end
    [r,q]=size(Data);
    fix_the_model=r;
    
    if ensemble==0
        fix_the_model=r;
        kprune=3*10^(-1);
        kfs=0.01;
        vigilance=9.9*10^(-1);
        paramet(1)=kprune;
        paramet(2)=kfs;
        paramet(3)=vigilance;
        demo='n';
        mode='c';
        drift=2;
        type_feature_weighting=7;
        % pclass_local_mod_improved_feature_weighting_multivariate4
        % the first time create the model
        
        [Weight,Center,Spread,rule,y,error,rules_significance,rules_novelty,datum_novelty,age,input_significance,population,time_index,born,time,classification_rate_testing,normalized_out,feature_weights,population_class_cluster,focalpoints,sigmapoints]=pclass_local_mod_improved_feature_weighting_multivariate4(Data,fix_the_model,paramet,demo,ninput,mode,drift,type_feature_weighting);
        buffer=Data;
        [v,vv]=size(Center);
        network_parameters=v*subset+(subset)^(2)*v+(subset+1)*v*noutput;
        network=struct('Center',Center,'Spread',Spread,'Weight',Weight,'ensemble_weight',1,'feature_weights',feature_weights,'network_parameters',network_parameters,'fuzzy_rule',v,'CR',classification_rate_testing,'focalpoints',focalpoints,'sigmapoints',sigmapoints,'population',population,'population_class_cluster',population_class_cluster);
        ensemble=ensemble+1;
        ensemblesize(1)=1;
        for k3=1:noutput
            error(:,k3,1)=0;
        end
    else
        start_test = tic;
        counter=counter+1;
        %% obtain prequential error
        Datatest=Data;
        ensembleoutputtest=zeros(size(Datatest,1),1);
        individualoutputtest=zeros(size(Datatest,1),size(network,1));
        misclassification=0;
        outens=[];
        sigma = zeros(size(Datatest,1),noutput);
        for k1=1:size(Datatest,1)
            stream=Datatest(k1,:);
            output=zeros(1,noutput);
            for m=1:ensemble
                weighted_stream=network(m).feature_weights.*stream(1:ninput);
                xek = [1, weighted_stream]';
                [nrule,ndimension]=size(network(m).Center);
                
                di=zeros(nrule,1);
                for k2=1:nrule
                    dis=(weighted_stream-network(m).Center(k2,:))*network(m).Spread(:,:,k2)*(weighted_stream-network(m).Center(k2,:))';
                    di(k2)=exp(-0.5*dis);
                end
                fsig=di/sum(di);
                for k2=1:nrule
                    Psik3((k2-1)*(ndimension+1)+1:k2*(ndimension+1),1) = fsig(k2)*xek;
                end
                
                ysem=Psik3'*network(m).Weight;
                
                [maxout,classlabel]=max(ysem);
                
                individualoutputtest(k1,m)=classlabel;
                output(classlabel)=output(classlabel)+network(m).ensemble_weight;
                sigma(k1,:) = sigma(k1,:) + network(m).ensemble_weight*ysem;
                clear Psik3
                
            end
            clear weightperrule ysem
            [maxout1,trueclasslabel]=max(stream(ndimension+1:end));
            [maxout,ensemblelabel]=max(sigma(k1,:));
            r_e(k1) = abs(1-maxout);
            ensembleoutputtest(k1)=ensemblelabel;
            if trueclasslabel==ensemblelabel
                misclassification=misclassification+1;
            end
        end
        totalrule=0;
        totalparameters=0;
        for m=1:size(network,1)
            totalrule=totalrule+network(m).fuzzy_rule;
            totalparameters=totalparameters+network(m).network_parameters;
        end
        
        B(counter)=(misclassification)/size(Datatest,1);
        C(counter)=totalrule;
        D(counter)=totalparameters;
        E(counter)=size(network,1);
        Residual_error(length(r_e)*t+(1-length(r_e)):length(r_e)*t,1) = r_e';
        test_time(t) = toc(start_test);
        
        %% training
        start_train = tic;
        traceinputweight=[];
        inputexpectation=mean(Data(:,1:ninput));
        inputvariance=var(Data(:,1:ninput));
        temporary=zeros(chunk_size,ninput);
        [upperbound,upperboundlocation]=max(Data(:,1:ninput));
        [lowerbound,lowerboundlocation]=min(Data(:,1:ninput));
        for iter=1:size(Data,1)
            for iter1=1:ninput
                temporary(iter,iter1)=Data(iter,iter1)-inputexpectation(iter1);
            end
        end
        for k1=1:size(Data,1)
            stream=Data(k1,:);
            xek = [1, stream(1:ninput)]';
            if inputpruning==1
                weight_input1=ones(1,size(Data,2));
                if partial==1
                    z_t=random('Binomial',1,0.2);
                    if z_t==1
                        c_t=randperm(ninput);
                        selectedinput=c_t(1:subset);
                        weight_input1(~selectedinput)=0;
                        stream=weight_input1.*stream;
                        traceinputweight(k1,:)=weight_input1(1:ninput);
                    else
                        traceinputweight(k1,:)=ones(1,ninput);
                    end
                end
                centeroverall=[];
                spreadoverall=[];
                weightoverall=[];
                for m=1:ensemble
                    centeroverall=[centeroverall;network(m).Center];
                    for i=1:size(network(m).Center,1)
                        spreadoverall(:,:,end+i)=network(m).Spread(:,:,i);
                    end
                    weightoverall=[weightoverall;network(m).Weight];
                end
                [totalrule,ndimension]=size(centeroverall);
                weight_input=zeros(ndimension,totalrule);
                
                di=zeros(totalrule,1);
                
                for k2=1:totalrule
                    dis=(stream(1:ninput)-centeroverall(k2,:))*spreadoverall(:,:,k2)*(stream(1:ninput)-centeroverall(k2,:))';
                    %  dis1=dis./spreadoverall(k2,:);
                    di(k2)=exp(-0.5*dis);
                end
                fsig=di/sum(di);
                
                for k2=1:totalrule
                    Psik1((k2-1)*(ndimension+1)+1:k2*(ndimension+1),1) = fsig(k2)*xek;
                end
                
                ysem=Psik1'*weightoverall;
                [maxout,classlabel]=max(ysem);
                [maxout1,trueclasslabel]=max(stream(ndimension+1:end));
                clear Psik1
                
                if classlabel~=trueclasslabel
                    weightperrule=zeros(ninput+1,noutput,totalrule);
                    weightperrule(:)=weightoverall;
                    for k2=1:totalrule
                        for out=1:noutput
                            if local==1
                                weightperrule(:,out,k2)=weightperrule(:,out,k2)-LR*RF*weightperrule(:,out,k2)-LR.*RF.*fsig(k2).*xek;
                            else
                                weightperrule(:,out,k2)=weightperrule(:,out,k2)-LR*RF*weightperrule(:,out,k2)-LR.*RF.*xek;
                            end
                            
                            if partial==1
                                weightperrule(:,out,k2)=weightperrule(:,out,k2)*min(1,10/(norm(weightperrule(:,out,k2))));
                            else
                                weightperrule(:,out,k2)=weightperrule(:,out,k2)*min(1,1/(sqrt(0.01)*norm(weightperrule(:,out,k2))));
                            end
                        end
                    end
                    for k2=1:totalrule
                        for j=1:ndimension
                            weight_input(j,k2)=sum(weightperrule(j,:,k2));
                        end
                    end
                    weight_input_total=zeros(1,ndimension);
                    for j=1:ndimension
                        weight_input_total(j)=sum(weight_input(j,:));
                    end
                    [values,index]=sort(abs(weight_input_total),'descend');
                    weight_input1=ones(1,size(stream,2));
                    weight_input1(index(subset+1:end))=0;
                    stream=stream.*weight_input1;
                    traceinputweight(k1,:)=weight_input1(1:ninput);
                end
            end
            output=zeros(1,noutput);
            pruning_list=[];
            for m=1:ensemble
                weighted_stream=network(m).feature_weights.*stream(1:ninput);
                xek = [1, weighted_stream]';
                [nrule,ndimension]=size(network(m).Center);
                if local==1
                    di=zeros(nrule,1);
                    for k2=1:nrule
                        dis=(weighted_stream-network(m).Center(k2,:))*network(m).Spread(:,:,k2)*(weighted_stream-network(m).Center(k2,:))';
                        %   dis1=dis./network(m).Spread(k2,:);
                        di(k2)=exp(-0.5*dis);
                    end
                    fsig=di/sum(di);
                    for k2=1:nrule
                        Psik2((k2-1)*(ndimension+1)+1:k2*(ndimension+1),1) = fsig(k2)*xek;
                    end
                    ysem=Psik2'*network(m).Weight;
                    [maxout,classlabel]=max(ysem);
                    [maxout1,trueclasslabel]=max(stream(ndimension+1:end));
                else
                    [nrule,ndimension]=size(network(m).Center);
                    weightperrule=zeros(ninput+1,noutput,nrule);
                    weightperrule(:)=network(m).Weight;
                    di=zeros(nrule,noutput);
                    for k2=1:nrule
                        for out=1:noutput
                            di(k2,out)=xek'*weightperrule(:,out,k2);
                        end
                    end
                    
                    ysem=sum(di)/sum(sum(di));
                    [maxout,classlabel]=max(ysem);
                    [maxout1,trueclasslabel]=max(stream(ndimension+1:end));
                    
                end
                for out=ninput+1:size(stream,2)
                    error(k1,out-ninput,m) = ysem(out-ninput) - stream(out);
                end
                Remp=zeros(1,noutput);
                
                for out=1:noutput
                    Remp(out)=sumsqr(error(:,out,m))/k1;
                end
                if classlabel~=trueclasslabel
                    network(m).ensemble_weight=network(m).ensemble_weight*decreasingfactor;
                else
                    network(m).ensemble_weight=min(network(m).ensemble_weight*(2-decreasingfactor),1);
                end
                output(classlabel)=output(classlabel)+network(m).ensemble_weight;
                clear Psik2
            end
            clear weightperrule ysem
            [maxout,ensemblelabel]=max(output);
            ensembleoutput(k1)=ensemblelabel;
            ensemblesize(k1)=ensemble;
            if mod(k1,p)==0
                
                
                if ensemblepruning1==1 && k1==size(Data,1)
                    SSM=zeros(1,ensemble);
                    a=zeros(1,ensemble);
                    pruning_list=[];
                    a=0;
                    for m=1:ensemble
                        a=a+network(m).ensemble_weight;
                    end
                    
                    for m=1:ensemble
                        activation=0;
                        network(m).ensemble_weight=network(m).ensemble_weight/a;
                        if a<0.01
                            network(m).ensemble_weight=network(m).ensemble_weight/a;
                        else
                            network(m).ensemble_weight=network(m).ensemble_weight;
                        end
                        
                        
                        if network(m).ensemble_weight<threshold && ensemble>1 && length(pruning_list)<size(network,1)
                            pruning_list=[pruning_list m];
                        end
                    end
                    if length(pruning_list)>=size(network,1)
                        pruning_list(end)=[];
                    end
                    if isempty(pruning_list)==false
                        network(pruning_list,:)=[];
                        error(:,:,pruning_list)=[];
                        RSMnew(pruning_list)=[];
                        RSMdev(pruning_list)=[];
                        ensemble=size(network,1);
                        ensemblesize(k1)=ensemble;
                        activation=1;
                    end
                end
                if ensemblepruning2==1 && ensemble>1 && isempty(pruning_list) && activation==0 && k1==size(Data,1)
                    pruning_list=[];
                    for m=1:size(network,1)
                        [nrule,ndimension]=size(network(m).Center);
                        euclidean=zeros(1,nrule);
                        euclideanALL=zeros(1,nrule);
                        varsj=zeros(ninput,nrule);
                        vi=zeros(1,nrule);
                        phi=zeros(ninput,nrule);
                        zeta=zeros(ninput,nrule);
                        expectationSJ=zeros(ninput,nrule);
                        widthTrans=zeros(ninput,nrule);
                        widthTrans1=zeros(1,nrule);
                        zeta1=zeros(1,nrule);
                        dist=zeros(1,ninput);
                        
                        temporary1=zeros(chunk_size,ninput);
                        pik=zeros(ninput+1,noutput,nrule);
                        pik(:)=network(m).Weight;
                        for k2=1:nrule
                            Mahalanobis=(stream(1:ninput)-network(m).Center(k2,:))*network(m).Spread(:,:,k2)*(stream(1:ninput)-network(m).Center(k2,:))';
                            radii=Mahalanobis*sqrt(diag(abs(network(m).Spread(:,:,k2))));
                            for k3=1:ninput
                                dist(k3)=stream(k3)-network(m).Center(k2,k3);
                                
                                varsj(k3,k2)=(mean(temporary(:,k3).^(4))-inputvariance(:,k3).^(2)+4.*inputvariance(:,k3).*(inputexpectation(:,k3)-network(m).Center(k2,k3)).^(2)+4.*mean(temporary(:,k3).^(3)).*(inputexpectation(:,k3)-network(m).Center(k2,k3)));
                                %dis1=mean(temporary.^(4))-inputvariance.^(2)+4.*inputvariance.*(inputexpectation-network(m).Center(k2,:)).^(2)+4.*mean(temporary.^(3)).*(inputexpectation-network(m).Center(k2,:));
                                expectationSJ(k3,k2)=(inputvariance(:,k3)+(stream(k3)-network(m).Center(k2,k3)).^(2));
                                
                            end
                            
                            expectationSJ1=sum(expectationSJ(:,k2));
                            varsj1=sum(varsj(:,k2));
                            for k4=1:ninput
                                a=xek'*pik(:,:,k2);
                                phi(k4,k2)=max(a)*exp((varsj1/radii(k4)^(2))-(expectationSJ1/radii(k4)));
                                if (varsj1/radii(k4)^(2))>(expectationSJ1/radii(k4))
                                    phi(k4,k2)=0.01;
                                end
                                zeta(k4,k2)=phi(k4,k2)/radii(k4)^(2);
                                if zeta(k4,k2)>1
                                    zeta(k4,k2)=0.01;
                                end
                                widthTrans(k4,k2)= (inputvariance(:,k4)+(inputexpectation(:,k4)-network(m).Center(k2,k4)).^(2))/radii(k4);
                            end
                            widthTrans1(k2)=prod(phi(:,k2))*sum(widthTrans(:,k2));
                            zeta1(k2)=sum(zeta(:,k2));
                        end
                        SSM(m)=1/3*sum(widthTrans1)+0.2/9*(ninput*sum(zeta1));
                        Remp=zeros(1,noutput);
                        
                        for out=1:noutput
                            Remp(out)=sumsqr(error(:,out,m))/k1;
                        end
                        RSMold=RSMnew;
                        RSMdevOLD=RSMdev;
                        RSM(m)=(prod(sqrt(Remp))+sqrt(SSM(m))+1)^(2)+sqrt(reallog(confidenceinterval)/(-2*chunk_size));
                        RSMnew(m)=((counter-1)/counter)*RSMold(m)+(RSM(m)/counter);
                        RSMdev(m)=((counter-1)/counter)*RSMdevOLD(m)+((RSM(m)-RSMnew(m))^2/(counter-1));
                        if RSM(m)<(RSMnew(m)-3*RSMdev(m)) && ensemble>1 && length(pruning_list)<size(network,1)
                            pruning_list=[pruning_list m];
                        end
                    end
                    if length(pruning_list)>=size(network,1)
                        pruning_list(end)=[];
                    end
                    if isempty(pruning_list)==false
                        network(pruning_list,:)=[];
                        error(:,:,pruning_list)=[];
                        ensemble=size(network,1);
                        RSMnew(pruning_list)=[];
                        RSMdev(pruning_list)=[];
                        ensemblesize(k1)=ensemble;
                    end
                end
                if k1==size(Data,1)
                    Zstat=mean(Data(:,1:ninput));
                    cuttingpoint=0;
                    for cut=1:size(Data,1)
                        Xstat=mean(Data(1:cut,1:ninput));
                        [Xupper,Xupper1]=max(Data(1:cut,1:ninput));
                        [Xlower,Xlower1]=min(Data(1:cut,1:ninput));
                        Xbound=(Xupper-Xlower)         *sqrt(((r-cut)/(2*cut*(r))*reallog(1/confidenceinterval)));
                        Zbound=(upperbound-lowerbound).*sqrt(((r-cut)/(2*cut*(r))*reallog(1/confidenceinterval)));
                        if mean(Xbound+Xstat)>=mean(Zstat+Zbound) & cut<r
                            cuttingpoint=cut;
                            Ystat=mean(Data(cuttingpoint+1:end,1:ninput));
                            [Yupper,Yupper1]=max(Data(cuttingpoint+1:end,1:ninput));
                            [Ylower,Ylower1]=min(Data(cuttingpoint+1:end,1:ninput));
                            Ybound =(Yupper-Ylower).*sqrt(((r-cuttingpoint)/(2*cuttingpoint*(r-cuttingpoint)))*reallog(1/lambdaD));
                            Ybound1=(Yupper-Ylower).*sqrt(((r-cuttingpoint)/(2*cuttingpoint*(r-cuttingpoint)))*reallog(1/lambdaW));
                            %      Ybound=(upperbound-lowerbound).*sqrt(((r-cut)/(2*cut*(r))*reallog(1/lambdaD)));
                            % Ybound1=(upperbound-lowerbound).*sqrt(((r-cut)/(2*cut*(r))*reallog(1/lambdaW)));
                            break
                            
                        end
                    end
                    if cuttingpoint==0
                        Ystat=Zstat;
                        Ybound=(upperbound-lowerbound).*sqrt(((r-cut)/(2*cut*(r))*reallog(1/lambdaD)));
                        Ybound1=(upperbound-lowerbound).*sqrt(((r-cut)/(2*cut*(r))*reallog(1/lambdaW)));
                    end
                    if (mean(abs(Xstat-Ystat)))>=mean(Ybound)
                        %% drift
                        
                        
                        kprune=3*10^(-1);
                        kfs=0.01;
                        vigilance=9.9*10^(-1);
                        paramet(1)=kprune;
                        paramet(2)=kfs;
                        paramet(3)=vigilance;
                        
                        
                        demo='n';
                        mode='c';
                        drift=2;
                        
                        type_feature_weighting=7;
                        
                        if isempty(buffer)
                            Data_fix=Data;
                            fix_the_model=r;
                        else
                            Data_fix=[buffer;Data];
                            fix_the_model=r+size(buffer,1);
                        end
                        [Weight,Center,Spread,rule,y,error,rules_significance,rules_novelty,datum_novelty,age,input_significance,population,time_index,born,time,classification_rate_testing,normalized_out,feature_weights,population_class_cluster,focalpoints,sigmapoints]=pclass_local_mod_improved_feature_weighting_multivariate4(Data_fix,fix_the_model,paramet,demo,ninput,mode,drift,type_feature_weighting);
                        [v,vv]=size(Center);
                        network_parameters=v*subset+(subset)^(2)*v+(subset+1)*v*noutput;
                        network=[network; struct('Center',Center,'Spread',Spread,'Weight',Weight,'ensemble_weight',1,'feature_weights',feature_weights,'network_parameters',network_parameters,'fuzzy_rule',v,'CR',classification_rate_testing,'focalpoints',focalpoints,'sigmapoints',sigmapoints,'population',population,'population_class_cluster',population_class_cluster)];
                        ensemble=size(network,1);
                        ensemblesize(k1)=ensemble;
                        RSMnew(ensemble)=0;
                        RSMdev(ensemble)=0;
                        for k3=1:noutput
                            error(:,k3,ensemble)=0;
                        end
                        buffer=[];
                    elseif (mean(abs(Xstat-Ystat)))>=mean(Ybound1) && (mean(abs(Xstat-Ystat)))<mean(Ybound)
                        %% Warning
                        buffer=[buffer;Data];
                        
                    else
                        %%stable
                        RMSE=zeros(noutput,size(network,1));
                        for m=1:size(network,1)
                            for out=1:noutput
                                RMSE(out,m)=sumsqr(error(:,out,m))/k1;
                            end
                        end
                        Rselected=zeros(1,size(network,1));
                        for m=1:size(network,1)
                            Rselected(m)=mean(RMSE(:,m));
                        end
                        [Rselected,index1]=min(Rselected);
                        fix_the_model=r;
                        
                        kprune=3*10^(-1);
                        kfs=0.01;
                        vigilance=9.9*10^(-1);
                        paramet(1)=kprune;
                        paramet(2)=kfs;
                        paramet(3)=vigilance;
                        demo='n';
                        mode='c';
                        drift=2;
                        type_feature_weighting=7;
                        buffer=[];
                        [Weight,Center,Spread,rule,y,error,rules_significance,rules_novelty,datum_novelty,input_significance,population,time,classification_rate_testing,normalized_out,feature_weights,population_class_cluster,focalpoints,sigmapoints]=pclass_multivariate_update1(Data,fix_the_model,paramet,demo,ninput,mode,drift,type_feature_weighting,network(index1));
                        [v,vv]=size(Center);
                        network_parameters=v*subset+(subset)^(2)*v+(subset+1)*v*noutput;
                        replacement=struct('Center',Center,'Spread',Spread,'Weight',Weight,'ensemble_weight',1,'feature_weights',feature_weights,'network_parameters',network_parameters,'fuzzy_rule',v,'CR',classification_rate_testing,'focalpoints',focalpoints,'sigmapoints',sigmapoints,'population',population,'population_class_cluster',population_class_cluster);
                        network(index1)=replacement;
                    end
                end
            end
            % % Start the model evolution (learning and prediction)
        end
        H(counter)=time;
        update_time(t) = toc(start_train);
    end
    %Datatest=Data(fix_the_model+1:end,:);
    
    
    
    
    
end
Clastestrat=mean(B);
Clastestdev=std(B);
TotRulerat=mean(C);
TotRuledev=std(C);
TotPamrat=mean(D);
TotPamdev=std(D);
SizeNetrat=mean(E);
SizeNetdev=std(E);
training_time = [mean(update_time) std(update_time)];
testing_time = [mean(test_time) std(test_time)];

ass2 = sprintf('results/%sresults.mat',name);
save(ass2,'Clastestrat','Clastestdev','TotRulerat','TotRuledev','TotPamrat','TotPamdev','SizeNetrat','SizeNetdev','training_time','testing_time','nFolds','chunk_size','Residual_error');
